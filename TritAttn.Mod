MODULE TritAttn;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Grouped-query attention over the KV cache";
	LIMITATIONS
		"TopK spreads 1/k over every score rather than the top k -- see ComputeWeights."
*)

IMPORT Math, TritIO, TritConfig, TritPacker, TritLinear, TritOps, Streams;

TYPE
	Attention* = OBJECT
	VAR
		emb, heads, kvHeads, headDim, kvDim, group: SIZE;
		invScale, theta: FLOAT32;
		mode: SIGNED32;
		topK: SIZE;

		Q-, K-, V-, Out-: TritLinear.Linear;

		PROCEDURE &Init* (cfg: TritConfig.Config);
		VAR qkv, outPlanes: SIZE;
		BEGIN
			emb := cfg.embeddingDim; heads := cfg.numHeads; kvHeads := cfg.numKVHeads;
			ASSERT(emb MOD heads = 0);
			ASSERT(heads MOD kvHeads = 0);
			headDim := emb DIV heads; kvDim := kvHeads * headDim; group := heads DIV kvHeads;
			invScale := 1 / Math.sqrt(FLOAT32(headDim));
			theta := cfg.ropeTheta; mode := cfg.attentionMode; topK := cfg.topK;

			qkv := cfg.attentionQkvPlanes; outPlanes := cfg.attentionOutPlanes;

			(* zeroed: every caller loads a file or imports trained weights straight
			   after, so the C# original's random initialisation was work thrown away *)
			NEW(Q, TritPacker.NewEmpty(emb, emb, qkv, cfg.groupSize, cfg.scaleMode), FALSE);
			NEW(K, TritPacker.NewEmpty(kvDim, emb, qkv, cfg.groupSize, cfg.scaleMode), FALSE);
			NEW(V, TritPacker.NewEmpty(kvDim, emb, qkv, cfg.groupSize, cfg.scaleMode), FALSE);
			NEW(Out, TritPacker.NewEmpty(emb, emb, outPlanes, cfg.groupSize, cfg.scaleMode), FALSE)
		END Init;

		PROCEDURE Forward* (input: TritIO.Floats; seq: SIZE; cache: TritOps.KVCache;
			VAR out: TritIO.Floats): BOOLEAN;
		VAR q, k, v, context, scores, weights: TritIO.Floats;
			startPos, i, h, kvh, qBase, kBase, absPos, j, d: SIZE; dot, acc: FLOAT32;
		BEGIN
			ASSERT(LEN(input) >= seq * emb);

			q := Q.ForwardSequence(input, seq);
			k := K.ForwardSequence(input, seq);
			v := V.ForwardSequence(input, seq);

			startPos := cache.length;
			TritOps.ApplyRoPE(q, k, seq, startPos, heads, kvHeads, headDim, theta);
			IF ~cache.Append(k, v, seq) THEN RETURN FALSE END;

			NEW(context, seq * emb);
			FOR i := 0 TO seq - 1 DO
				absPos := startPos + i;
				FOR h := 0 TO heads - 1 DO
					kvh := h DIV group;
					qBase := i * emb + h * headDim;

					NEW(scores, absPos + 1);
					FOR j := 0 TO absPos DO
						kBase := j * kvDim + kvh * headDim;
						dot := 0;
						FOR d := 0 TO headDim - 1 DO
							dot := dot + q[qBase + d] * cache.K[kBase + d]
						END;
						scores[j] := dot * invScale
					END;

					weights := ComputeWeights(scores);

					FOR d := 0 TO headDim - 1 DO
						acc := 0;
						FOR j := 0 TO absPos DO
							acc := acc + weights[j] * cache.V[j * kvDim + kvh * headDim + d]
						END;
						context[qBase + d] := acc
					END
				END
			END;

			out := Out.ForwardSequence(context, seq);
			RETURN TRUE
		END Forward;

		PROCEDURE ComputeWeights (scores: TritIO.Floats): TritIO.Floats;
		VAR weights: TritIO.Floats; n, i, best, k: SIZE; maxScore, sum, inv, uniform, w: FLOAT32;
		BEGIN
			n := LEN(scores);
			NEW(weights, n);
			IF n = 0 THEN RETURN weights END;

			IF mode = TritConfig.AttnHardmax THEN
				best := 0;
				FOR i := 1 TO n - 1 DO
					IF scores[i] > scores[best] THEN best := i END
				END;
				weights[best] := 1

			ELSIF mode = TritConfig.AttnTopK THEN
				k := topK;
				IF k < 1 THEN k := 1 END;
				IF k > n THEN k := n END;
				(* ponytail: spreads 1/k over every score, not just the top k. That is
				   the C# original's behaviour, kept so the port is never a silent model
				   change; fix it when TopK stops being an experiment. *)
				w := 1 / FLOAT32(k);
				FOR i := 0 TO n - 1 DO weights[i] := w END

			ELSE
				maxScore := scores[0];
				FOR i := 1 TO n - 1 DO
					IF scores[i] > maxScore THEN maxScore := scores[i] END
				END;
				sum := 0;
				FOR i := 0 TO n - 1 DO
					weights[i] := Math.exp(scores[i] - maxScore);
					sum := sum + weights[i]
				END;
				IF sum <= 1.0E-12 THEN
					uniform := 1 / FLOAT32(n);
					FOR i := 0 TO n - 1 DO weights[i] := uniform END
				ELSE
					inv := 1 / (sum + 1.0E-10);
					FOR i := 0 TO n - 1 DO weights[i] := weights[i] * inv END
				END
			END;
			RETURN weights
		END ComputeWeights;

		PROCEDURE Save* (w: Streams.Writer);
		BEGIN Q.Save(w); K.Save(w); V.Save(w); Out.Save(w)
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN
			RETURN Q.Load(r, message) & K.Load(r, message) & V.Load(r, message) & Out.Load(r, message)
		END Load;

	END Attention;

PROCEDURE {TEST} TestSoftmaxWeightsSumToOne*;
VAR cfg: TritConfig.Config; a: Attention; scores, w: TritIO.Floats; i: SIZE; sum: FLOAT32;
BEGIN
	NEW(cfg); cfg.embeddingDim := 8; cfg.numHeads := 2; cfg.numKVHeads := 1;
	cfg.vocabSize := 16; cfg.hiddenDim := 8; cfg.numLayers := 1; cfg.groupSize := 8;
	cfg.SetUniformPlanes(1);
	NEW(a, cfg);

	NEW(scores, 4);
	scores[0] := 1.0; scores[1] := 3.0; scores[2] := -2.0; scores[3] := 0.5;
	w := a.ComputeWeights(scores);

	sum := 0;
	FOR i := 0 TO 3 DO ASSERT(w[i] > 0); sum := sum + w[i] END;
	ASSERT(ABS(sum - 1.0) < 0.001);
	ASSERT(w[1] > w[0]);					(* the largest score takes the most weight *)
	ASSERT(w[2] < w[3])
END TestSoftmaxWeightsSumToOne;

PROCEDURE {TEST} TestCausalShapeAndCache*;
VAR cfg: TritConfig.Config; a: Attention; cache: TritOps.KVCache; x, out: TritIO.Floats; i: SIZE;
BEGIN
	NEW(cfg); cfg.embeddingDim := 8; cfg.numHeads := 2; cfg.numKVHeads := 1;
	cfg.vocabSize := 16; cfg.hiddenDim := 8; cfg.numLayers := 1; cfg.groupSize := 8;
	cfg.maxSeqLen := 4; cfg.SetUniformPlanes(1);
	NEW(a, cfg);
	NEW(cache, cfg.maxSeqLen, cfg.KVDim());

	NEW(x, 3 * cfg.embeddingDim);
	FOR i := 0 TO LEN(x) - 1 DO x[i] := (i MOD 5) - 2 END;

	ASSERT(a.Forward(x, 3, cache, out));
	ASSERT(LEN(out) = 3 * cfg.embeddingDim);
	ASSERT(cache.length = 3);				(* the cache advanced by the whole prefix *)

	NEW(x, cfg.embeddingDim);
	ASSERT(a.Forward(x, 1, cache, out));
	ASSERT(cache.length = 4);
	ASSERT(~a.Forward(x, 1, cache, out))	(* past maxSeqLen: reported, not trapped *)
END TestCausalShapeAndCache;

END TritAttn.
