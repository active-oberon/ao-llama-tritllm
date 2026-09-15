MODULE TritTrainBlock;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "One trainable Llama layer: FP32 weights, quantized on every forward pass";
	LIMITATIONS
		"the grouped-query attention backward is written out by hand rather than
		 composed from autograd operations -- a node per head per position would be
		 a graph the size of the sequence squared. TestAttentionGradient is what
		 holds it to the finite differences."
*)

IMPORT Math, Random, TritIO, TritConfig, TritTensor, TritGrad;

TYPE
	Node = TritTensor.Node;
	FloatsArray = POINTER TO ARRAY OF TritIO.Floats;

	(** AttentionNode is causal multi-head attention with grouped keys and values. *)
	AttentionNode = OBJECT (TritTensor.Node)
	VAR
		q, k, v: Node;
		seq, emb, kvDim, heads, kvHeads, headDim, group: SIZE;
		scale: FLOAT32;
		qHeads, kHeads, vHeads, attnW: FloatsArray;

		PROCEDURE Backward*;
		VAR dKHeads, dVHeads: FloatsArray;
			h, kvh, hStart, i, j, d, kv: SIZE;
			dAttnW, dV, dQHead, dKHead, w: TritIO.Floats;
			g, dot, ds: FLOAT32;
		BEGIN
			NEW(dKHeads, kvHeads); NEW(dVHeads, kvHeads);
			FOR kv := 0 TO kvHeads - 1 DO
				NEW(dKHeads[kv], seq * headDim); NEW(dVHeads[kv], seq * headDim)
			END;

			FOR h := 0 TO heads - 1 DO
				kvh := h DIV group; hStart := h * headDim;
				w := attnW[h];

				NEW(dAttnW, seq * seq); NEW(dV, seq * headDim);

				FOR i := 0 TO seq - 1 DO
					FOR d := 0 TO headDim - 1 DO
						g := grad[i * emb + hStart + d];
						FOR j := 0 TO i DO
							dAttnW[i * seq + j] := dAttnW[i * seq + j] + g * vHeads[kvh][j * headDim + d];
							dV[j * headDim + d] := dV[j * headDim + d] + g * w[i * seq + j]
						END
					END
				END;

				(* through the softmax, and the 1/sqrt(headDim) the scores carried *)
				FOR i := 0 TO seq - 1 DO
					dot := 0;
					FOR j := 0 TO i DO dot := dot + w[i * seq + j] * dAttnW[i * seq + j] END;
					FOR j := 0 TO i DO
						dAttnW[i * seq + j] := w[i * seq + j] * (dAttnW[i * seq + j] - dot) * scale
					END
				END;

				NEW(dQHead, seq * headDim); NEW(dKHead, seq * headDim);
				FOR i := 0 TO seq - 1 DO
					FOR j := 0 TO i DO
						ds := dAttnW[i * seq + j];
						FOR d := 0 TO headDim - 1 DO
							dQHead[i * headDim + d] := dQHead[i * headDim + d] + ds * kHeads[kvh][j * headDim + d];
							dKHead[j * headDim + d] := dKHead[j * headDim + d] + ds * qHeads[h][i * headDim + d]
						END
					END
				END;

				IF q.grad # NIL THEN
					FOR i := 0 TO seq - 1 DO
						FOR d := 0 TO headDim - 1 DO
							q.grad[i * emb + hStart + d] := q.grad[i * emb + hStart + d] + dQHead[i * headDim + d]
						END
					END
				END;

				FOR i := 0 TO seq * headDim - 1 DO
					dKHeads[kvh][i] := dKHeads[kvh][i] + dKHead[i];
					dVHeads[kvh][i] := dVHeads[kvh][i] + dV[i]
				END
			END;

			IF k.grad # NIL THEN
				FOR kv := 0 TO kvHeads - 1 DO
					FOR i := 0 TO seq - 1 DO
						FOR d := 0 TO headDim - 1 DO
							k.grad[i * kvDim + kv * headDim + d] :=
								k.grad[i * kvDim + kv * headDim + d] + dKHeads[kv][i * headDim + d]
						END
					END
				END
			END;

			IF v.grad # NIL THEN
				FOR kv := 0 TO kvHeads - 1 DO
					FOR i := 0 TO seq - 1 DO
						FOR d := 0 TO headDim - 1 DO
							v.grad[i * kvDim + kv * headDim + d] :=
								v.grad[i * kvDim + kv * headDim + d] + dVHeads[kv][i * headDim + d]
						END
					END
				END
			END
		END Backward;

	END AttentionNode;

	Block* = OBJECT
	VAR
		norm1Gamma-, norm2Gamma-: Node;
		qWeight-, kWeight-, vWeight-, outWeight-: Node;
		ffnGateWeight-, ffnUpWeight-, ffnDownWeight-: Node;

		emb, heads, kvHeads, headDim, hidden: SIZE;

		PROCEDURE &Init* (emb, heads, kvHeads, hidden: SIZE);
		VAR kvDim: SIZE; std, stdDown: FLOAT32; rng: Random.Sequence;
		BEGIN
			ASSERT(emb MOD heads = 0);
			ASSERT(heads MOD kvHeads = 0);
			SELF.emb := emb; SELF.heads := heads; SELF.kvHeads := kvHeads;
			headDim := emb DIV heads; SELF.hidden := hidden;
			kvDim := kvHeads * headDim;

			std := Math.sqrt(2.0 / emb);
			stdDown := Math.sqrt(2.0 / hidden);

			NEW(rng); rng.InitSeed(123);

			norm1Gamma := TritTensor.Ones(emb, TRUE);
			qWeight := Normal(rng, emb, emb, std);
			kWeight := Normal(rng, kvDim, emb, std);
			vWeight := Normal(rng, kvDim, emb, std);
			outWeight := Normal(rng, emb, emb, std);

			norm2Gamma := TritTensor.Ones(emb, TRUE);
			ffnGateWeight := Normal(rng, hidden, emb, std);
			ffnUpWeight := Normal(rng, hidden, emb, std);
			ffnDownWeight := Normal(rng, emb, hidden, stdDown)
		END Init;

		PROCEDURE Forward* (x: Node; cfg: TritConfig.Config): Node;
		VAR residual, normed, q, k, v, attn, attnOut, gate, up, hiddenNode, down: Node;
		BEGIN
			residual := x;
			normed := TritGrad.RmsNorm(x, norm1Gamma);

			q := TritGrad.MatMul(normed, TritGrad.Transpose(TritGrad.Quantize(qWeight, cfg.attentionQkvPlanes, cfg)));
			k := TritGrad.MatMul(normed, TritGrad.Transpose(TritGrad.Quantize(kWeight, cfg.attentionQkvPlanes, cfg)));
			v := TritGrad.MatMul(normed, TritGrad.Transpose(TritGrad.Quantize(vWeight, cfg.attentionQkvPlanes, cfg)));

			q := TritGrad.ApplyRoPE(q, cfg.numHeads, headDim, cfg.ropeTheta);
			k := TritGrad.ApplyRoPE(k, cfg.numKVHeads, headDim, cfg.ropeTheta);

			attn := Attention(q, k, v, cfg.numHeads, cfg.numKVHeads, headDim);
			attnOut := TritGrad.MatMul(attn,
				TritGrad.Transpose(TritGrad.Quantize(outWeight, cfg.attentionOutPlanes, cfg)));

			x := TritGrad.Add(residual, attnOut);

			residual := x;
			normed := TritGrad.RmsNorm(x, norm2Gamma);

			gate := TritGrad.MatMul(normed,
				TritGrad.Transpose(TritGrad.Quantize(ffnGateWeight, cfg.ffnGatePlanes, cfg)));
			up := TritGrad.MatMul(normed,
				TritGrad.Transpose(TritGrad.Quantize(ffnUpWeight, cfg.ffnUpPlanes, cfg)));

			hiddenNode := TritGrad.Mul(TritGrad.Silu(gate), up);
			down := TritGrad.MatMul(hiddenNode,
				TritGrad.Transpose(TritGrad.Quantize(ffnDownWeight, cfg.ffnDownPlanes, cfg)));

			RETURN TritGrad.Add(residual, down)
		END Forward;

		PROCEDURE Parameters* (VAR list: ARRAY OF Node; VAR n: SIZE);
		BEGIN
			list[n] := norm1Gamma; INC(n);
			list[n] := qWeight; INC(n);
			list[n] := kWeight; INC(n);
			list[n] := vWeight; INC(n);
			list[n] := outWeight; INC(n);
			list[n] := norm2Gamma; INC(n);
			list[n] := ffnGateWeight; INC(n);
			list[n] := ffnUpWeight; INC(n);
			list[n] := ffnDownWeight; INC(n)
		END Parameters;

	END Block;

PROCEDURE Normal* (rng: Random.Sequence; rows, cols: SIZE; std: FLOAT32): Node;
VAR data: TritIO.Floats; i: SIZE;
BEGIN
	NEW(data, rows * cols);
	FOR i := 0 TO LEN(data) - 1 DO data[i] := std * rng.Gaussian() END;
	RETURN TritTensor.New(data, rows, cols, TRUE)
END Normal;

(** Attention is causal, grouped-query, and softmaxed per row. *)
PROCEDURE Attention* (q, k, v: Node; heads, kvHeads, headDim: SIZE): Node;
VAR n: AttentionNode; out: TritIO.Floats; seq, emb, kvDim, group, h, kv, s, i, j, d, kvh: SIZE;
	scale, maxScore, sum, inv, dot, acc: FLOAT32; w: TritIO.Floats;
BEGIN
	seq := q.rows; emb := q.cols; kvDim := k.cols;
	group := heads DIV kvHeads;
	scale := 1 / Math.sqrt(FLOAT32(headDim));

	NEW(n);
	NEW(out, seq * emb);
	NEW(n.qHeads, heads); NEW(n.kHeads, kvHeads); NEW(n.vHeads, kvHeads);
	NEW(n.attnW, heads);

	FOR h := 0 TO heads - 1 DO
		NEW(n.qHeads[h], seq * headDim);
		FOR s := 0 TO seq - 1 DO
			FOR d := 0 TO headDim - 1 DO
				n.qHeads[h][s * headDim + d] := q.data[s * emb + h * headDim + d]
			END
		END
	END;
	FOR kv := 0 TO kvHeads - 1 DO
		NEW(n.kHeads[kv], seq * headDim); NEW(n.vHeads[kv], seq * headDim);
		FOR s := 0 TO seq - 1 DO
			FOR d := 0 TO headDim - 1 DO
				n.kHeads[kv][s * headDim + d] := k.data[s * kvDim + kv * headDim + d];
				n.vHeads[kv][s * headDim + d] := v.data[s * kvDim + kv * headDim + d]
			END
		END
	END;

	FOR h := 0 TO heads - 1 DO
		kvh := h DIV group;
		NEW(w, seq * seq);

		FOR i := 0 TO seq - 1 DO
			maxScore := MIN(FLOAT32);
			FOR j := 0 TO i DO
				dot := 0;
				FOR d := 0 TO headDim - 1 DO
					dot := dot + n.qHeads[h][i * headDim + d] * n.kHeads[kvh][j * headDim + d]
				END;
				w[i * seq + j] := dot * scale;
				IF w[i * seq + j] > maxScore THEN maxScore := w[i * seq + j] END
			END;

			sum := 0;
			FOR j := 0 TO i DO
				w[i * seq + j] := Math.exp(w[i * seq + j] - maxScore);
				sum := sum + w[i * seq + j]
			END;
			inv := 1 / (sum + 1.0E-10);
			FOR j := 0 TO i DO w[i * seq + j] := w[i * seq + j] * inv END
		END;

		n.attnW[h] := w;

		FOR i := 0 TO seq - 1 DO
			FOR d := 0 TO headDim - 1 DO
				acc := 0;
				FOR j := 0 TO i DO acc := acc + w[i * seq + j] * n.vHeads[kvh][j * headDim + d] END;
				out[i * emb + h * headDim + d] := acc
			END
		END
	END;

	TritTensor.Setup(n, out, seq, emb, q.requiresGrad OR k.requiresGrad OR v.requiresGrad);
	n.q := q; n.k := k; n.v := v;
	n.seq := seq; n.emb := emb; n.kvDim := kvDim;
	n.heads := heads; n.kvHeads := kvHeads; n.headDim := headDim; n.group := group;
	n.scale := scale;
	n.AddChild(q); n.AddChild(k); n.AddChild(v);
	RETURN n
END Attention;

(* ---- the check that matters: the hand-written backward against a difference ---- *)

PROCEDURE Filled (n, seed: SIZE): TritIO.Floats;
VAR v: TritIO.Floats; i: SIZE;
BEGIN
	NEW(v, n);
	FOR i := 0 TO n - 1 DO v[i] := ((i * 5 + seed * 17) MOD 13) / 6.0 - 1.0 END;
	RETURN v
END Filled;

PROCEDURE Copy (src: TritIO.Floats): TritIO.Floats;
VAR out: TritIO.Floats; i: SIZE;
BEGIN
	NEW(out, LEN(src));
	FOR i := 0 TO LEN(src) - 1 DO out[i] := src[i] END;
	RETURN out
END Copy;

PROCEDURE {TEST} TestAttentionGradient*;
VAR qData, kData, vData, weight: TritIO.Floats; q, k, v, out: Node;
	i, probe: SIZE; up, down, numeric, analytic, tolerance: FLOAT32;
CONST seq = 4; heads = 2; kvHeads = 1; headDim = 4;
	emb = heads * headDim; kvDim = kvHeads * headDim;

	(* the loss is sum(out * weight), so dL/dout is just weight *)
	PROCEDURE Loss (): FLOAT32;
	VAR a, b, c, o: Node; sum: FLOAT32; j: SIZE;
	BEGIN
		a := TritTensor.New(Copy(qData), seq, emb, TRUE);
		b := TritTensor.New(Copy(kData), seq, kvDim, TRUE);
		c := TritTensor.New(Copy(vData), seq, kvDim, TRUE);
		o := Attention(a, b, c, heads, kvHeads, headDim);
		sum := 0;
		FOR j := 0 TO LEN(o.data) - 1 DO sum := sum + o.data[j] * weight[j] END;
		RETURN sum
	END Loss;

	PROCEDURE Check (data, grad: TritIO.Floats);
	VAR p, idx: SIZE;
	BEGIN
		FOR p := 0 TO 2 DO
			idx := p * (LEN(data) DIV 3);
			data[idx] := data[idx] + 0.01; up := Loss();
			data[idx] := data[idx] - 0.02; down := Loss();
			data[idx] := data[idx] + 0.01;
			numeric := (up - down) / 0.02;
			analytic := grad[idx];
			tolerance := ABS(analytic);
			IF tolerance < 1 THEN tolerance := 1 END;
			ASSERT(ABS(numeric - analytic) / tolerance <= 0.02)
		END
	END Check;

BEGIN
	qData := Filled(seq * emb, 1);
	kData := Filled(seq * kvDim, 2);
	vData := Filled(seq * kvDim, 3);
	weight := Filled(seq * emb, 4);

	q := TritTensor.New(qData, seq, emb, TRUE);
	k := TritTensor.New(kData, seq, kvDim, TRUE);
	v := TritTensor.New(vData, seq, kvDim, TRUE);
	out := Attention(q, k, v, heads, kvHeads, headDim);

	FOR i := 0 TO LEN(out.grad) - 1 DO out.grad[i] := weight[i] END;
	out.Backward;

	Check(qData, q.grad);
	Check(kData, k.grad);
	Check(vData, v.grad)
END TestAttentionGradient;

PROCEDURE {TEST} TestCausalMask*;
VAR qData, kData, vData: TritIO.Floats; q, k, v, out: Node; before: TritIO.Floats; d: SIZE;
CONST seq = 3; heads = 1; kvHeads = 1; headDim = 4; emb = 4;
BEGIN
	qData := Filled(seq * emb, 1);
	kData := Filled(seq * emb, 2);
	vData := Filled(seq * emb, 3);

	q := TritTensor.New(qData, seq, emb, FALSE);
	k := TritTensor.New(kData, seq, emb, FALSE);
	v := TritTensor.New(Copy(vData), seq, emb, FALSE);
	out := Attention(q, k, v, heads, kvHeads, headDim);

	NEW(before, emb);
	FOR d := 0 TO emb - 1 DO before[d] := out.data[d] END;

	(* position 0 may not see position 2: change the future and it must not move *)
	vData[2 * emb] := vData[2 * emb] + 100.0;
	v := TritTensor.New(Copy(vData), seq, emb, FALSE);
	out := Attention(q, k, v, heads, kvHeads, headDim);
	FOR d := 0 TO emb - 1 DO ASSERT(out.data[d] = before[d]) END;
	ASSERT(out.data[2 * emb] # before[0])
END TestCausalMask;

END TritTrainBlock.
