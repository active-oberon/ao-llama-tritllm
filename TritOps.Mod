MODULE TritOps;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Inference-side activations, RoPE and the KV cache -- all FP32";
	LIMITATIONS
		"activations are never quantized by default: only the weights are ternary,
		 and that is what puts the model file in kilobytes."
*)

IMPORT Math, TritIO, TritPacker;

CONST
	(** NegInfinity masks a logit or a score. A finite sentinel rather than a real
		infinity: it divides by a temperature and exponentiates without trapping. *)
	NegInfinity* = -1.0E30;

TYPE
	(** LayerWeights carries dense FP32 weights from the trainable model into the packed one. *)
	LayerWeights* = RECORD
		norm1*, norm2*: TritIO.Floats;
		q*, k*, v*, out*: TritIO.Floats;
		gate*, up*, down*: TritIO.Floats
	END;
	LayerWeightsArray* = POINTER TO ARRAY OF LayerWeights;

	(** KVCache holds the keys and values already computed for this conversation. *)
	KVCache* = OBJECT
	VAR
		maxSeq-, kvDim-, length-: SIZE;
		K-, V-: TritIO.Floats;

		PROCEDURE &Init* (maxSeq, kvDim: SIZE);
		BEGIN
			SELF.maxSeq := maxSeq; SELF.kvDim := kvDim; length := 0;
			NEW(K, maxSeq * kvDim); NEW(V, maxSeq * kvDim)
		END Init;

		PROCEDURE Reset*;
		BEGIN length := 0
		END Reset;

		PROCEDURE Append* (k, v: TritIO.Floats; seq: SIZE): BOOLEAN;
		VAR i, base: SIZE;
		BEGIN
			IF seq <= 0 THEN RETURN TRUE END;
			IF length + seq > maxSeq THEN RETURN FALSE END;
			base := length * kvDim;
			FOR i := 0 TO seq * kvDim - 1 DO K[base + i] := k[i]; V[base + i] := v[i] END;
			INC(length, seq);
			RETURN TRUE
		END Append;

	END KVCache;

PROCEDURE Ones* (n: SIZE): TritIO.Floats;
VAR v: TritIO.Floats; i: SIZE;
BEGIN
	NEW(v, n);
	FOR i := 0 TO n - 1 DO v[i] := 1 END;
	RETURN v
END Ones;

PROCEDURE Add* (a, b: TritIO.Floats; count: SIZE): TritIO.Floats;
VAR out: TritIO.Floats; i: SIZE;
BEGIN
	NEW(out, count);
	FOR i := 0 TO count - 1 DO out[i] := a[i] + b[i] END;
	RETURN out
END Add;

PROCEDURE RmsNorm* (input: TritIO.Floats; rows, dim: SIZE; gamma: TritIO.Floats): TritIO.Floats;
VAR out: TritIO.Floats; r, i, offset: SIZE; meanSq, inv, g, x: FLOAT32;
BEGIN
	NEW(out, rows * dim);
	FOR r := 0 TO rows - 1 DO
		offset := r * dim;
		meanSq := 0;
		FOR i := 0 TO dim - 1 DO x := input[offset + i]; meanSq := meanSq + x * x END;
		meanSq := meanSq / dim;
		inv := 1 / Math.sqrt(meanSq + 1.0E-5);
		FOR i := 0 TO dim - 1 DO
			IF i < LEN(gamma) THEN g := gamma[i] ELSE g := 1 END;
			out[offset + i] := input[offset + i] * inv * g
		END
	END;
	RETURN out
END RmsNorm;

(** Requantize rounds activations onto the ternary grid; off unless the config asks. *)
PROCEDURE Requantize* (data: TritIO.Floats; rows, dim, planes, groupSize: SIZE;
	scaleMode: SIGNED32): TritIO.Floats;
VAR p: TritPacker.Packer;
BEGIN
	p := TritPacker.FromFloat(data, rows, dim, planes, groupSize, scaleMode, 0);
	RETURN p.ToFloat()
END Requantize;

PROCEDURE Silu* (x: TritIO.Floats);
VAR i: SIZE; v: FLOAT32;
BEGIN
	FOR i := 0 TO LEN(x) - 1 DO
		v := x[i];
		x[i] := v / (1 + Math.exp(-v))
	END
END Silu;

(** ApplyRoPE rotates q and k in place; startPos is how far the KV cache already is. *)
PROCEDURE ApplyRoPE* (q, k: TritIO.Floats; seq, startPos, qHeads, kvHeads, headDim: SIZE;
	theta: FLOAT32);
VAR invFreq: TritIO.Floats; half, qDim, kDim, d, pos, absPos, h, i1, i2, offset: SIZE;
	angle, cos, sin, x1, x2: FLOAT32;
BEGIN
	ASSERT(headDim MOD 2 = 0);
	half := headDim DIV 2; qDim := qHeads * headDim; kDim := kvHeads * headDim;

	NEW(invFreq, half);
	FOR d := 0 TO half - 1 DO
		invFreq[d] := 1 / Math.exp(2.0 * d / headDim * Math.ln(theta))
	END;

	FOR pos := 0 TO seq - 1 DO
		absPos := startPos + pos;
		FOR d := 0 TO half - 1 DO
			angle := absPos * invFreq[d];
			cos := Math.cos(angle); sin := Math.sin(angle);

			offset := pos * qDim;
			FOR h := 0 TO qHeads - 1 DO
				i1 := offset + h * headDim + d; i2 := i1 + half;
				x1 := q[i1]; x2 := q[i2];
				q[i1] := x1 * cos - x2 * sin;
				q[i2] := x1 * sin + x2 * cos
			END;

			offset := pos * kDim;
			FOR h := 0 TO kvHeads - 1 DO
				i1 := offset + h * headDim + d; i2 := i1 + half;
				x1 := k[i1]; x2 := k[i2];
				k[i1] := x1 * cos - x2 * sin;
				k[i2] := x1 * sin + x2 * cos
			END
		END
	END
END ApplyRoPE;

PROCEDURE {TEST} TestRmsNorm*;
VAR x, gamma, out: TritIO.Floats; i: SIZE; sum: FLOAT32;
BEGIN
	NEW(x, 8); NEW(gamma, 4);
	FOR i := 0 TO 7 DO x[i] := i + 1 END;
	FOR i := 0 TO 3 DO gamma[i] := 1 END;

	out := RmsNorm(x, 2, 4, gamma);
	(* each row is scaled to unit root-mean-square *)
	sum := 0;
	FOR i := 0 TO 3 DO sum := sum + out[i] * out[i] END;
	ASSERT(ABS(sum / 4 - 1.0) < 0.001);
	sum := 0;
	FOR i := 4 TO 7 DO sum := sum + out[i] * out[i] END;
	ASSERT(ABS(sum / 4 - 1.0) < 0.001)
END TestRmsNorm;

PROCEDURE {TEST} TestRoPEPreservesLength*;
VAR q, k: TritIO.Floats; i: SIZE; before, after: FLOAT32;
CONST seq = 3; heads = 2; headDim = 4;
BEGIN
	NEW(q, seq * heads * headDim); NEW(k, seq * headDim);
	FOR i := 0 TO LEN(q) - 1 DO q[i] := (i MOD 5) - 2 END;
	FOR i := 0 TO LEN(k) - 1 DO k[i] := (i MOD 3) - 1 END;

	before := 0;
	FOR i := 0 TO LEN(q) - 1 DO before := before + q[i] * q[i] END;

	ApplyRoPE(q, k, seq, 0, heads, 1, headDim, 10000.0);

	(* a rotation changes direction, never length *)
	after := 0;
	FOR i := 0 TO LEN(q) - 1 DO after := after + q[i] * q[i] END;
	ASSERT(ABS(after - before) < 0.001 * before + 0.001)
END TestRoPEPreservesLength;

PROCEDURE {TEST} TestKVCache*;
VAR c: KVCache; k, v: TritIO.Floats; i: SIZE;
BEGIN
	NEW(c, 4, 2);
	NEW(k, 4); NEW(v, 4);
	FOR i := 0 TO 3 DO k[i] := i; v[i] := -i END;

	ASSERT(c.Append(k, v, 2));
	ASSERT(c.length = 2);
	ASSERT(c.K[0] = 0); ASSERT(c.K[3] = 3);
	ASSERT(c.Append(k, v, 2));
	ASSERT(c.length = 4);
	ASSERT(~c.Append(k, v, 2));		(* overflow is reported, not trapped *)
	c.Reset;
	ASSERT(c.length = 0)
END TestKVCache;

END TritOps.
