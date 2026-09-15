MODULE TritBlock;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "One pre-norm Llama layer: norm, attention, norm, FFN, each with a residual"
*)

IMPORT Streams, TritIO, TritConfig, TritAttn, TritFFN, TritOps;

TYPE
	Block* = OBJECT
	VAR
		emb: SIZE;
		norm1-, norm2-: TritIO.Floats;
		attention-: TritAttn.Attention;
		ffn-: TritFFN.SwiGLU;

		quantizeActivations: BOOLEAN;
		activationPlanes, groupSize: SIZE;
		scaleMode: SIGNED32;

		PROCEDURE &Init* (cfg: TritConfig.Config);
		BEGIN
			emb := cfg.embeddingDim;
			norm1 := TritOps.Ones(emb); norm2 := TritOps.Ones(emb);
			NEW(attention, cfg); NEW(ffn, cfg);
			quantizeActivations := cfg.quantizeActivations;
			activationPlanes := cfg.activationPlanes;
			groupSize := cfg.groupSize; scaleMode := cfg.scaleMode
		END Init;

		PROCEDURE Forward* (input: TritIO.Floats; seq: SIZE; cache: TritOps.KVCache;
			VAR out: TritIO.Floats): BOOLEAN;
		VAR normed, attn, residual, ffnOut: TritIO.Floats;
		BEGIN
			normed := TritOps.RmsNorm(input, seq, emb, norm1);
			IF ~attention.Forward(normed, seq, cache, attn) THEN RETURN FALSE END;

			residual := TritOps.Add(input, attn, seq * emb);
			IF quantizeActivations THEN
				residual := TritOps.Requantize(residual, seq, emb, activationPlanes, groupSize, scaleMode)
			END;

			normed := TritOps.RmsNorm(residual, seq, emb, norm2);
			ffnOut := ffn.Forward(normed, seq);

			out := TritOps.Add(residual, ffnOut, seq * emb);
			IF quantizeActivations THEN
				out := TritOps.Requantize(out, seq, emb, activationPlanes, groupSize, scaleMode)
			END;
			RETURN TRUE
		END Forward;

		PROCEDURE Save* (w: Streams.Writer);
		BEGIN
			TritIO.WriteFloats(w, norm1); TritIO.WriteFloats(w, norm2);
			attention.Save(w); ffn.Save(w)
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN
			norm1 := TritIO.ReadFloats(r); norm2 := TritIO.ReadFloats(r);
			RETURN attention.Load(r, message) & ffn.Load(r, message)
		END Load;

	END Block;

END TritBlock.
