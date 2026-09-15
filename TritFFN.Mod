MODULE TritFFN;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "SwiGLU feed-forward: gate * up, then down"
*)

IMPORT Streams, TritIO, TritConfig, TritPacker, TritLinear, TritOps;

TYPE
	SwiGLU* = OBJECT
	VAR
		emb, hidden: SIZE;
		Gate-, Up-, Down-: TritLinear.Linear;

		quantizeActivations: BOOLEAN;
		activationPlanes, groupSize: SIZE;
		scaleMode: SIGNED32;

		PROCEDURE &Init* (cfg: TritConfig.Config);
		BEGIN
			emb := cfg.embeddingDim; hidden := cfg.hiddenDim;
			quantizeActivations := cfg.quantizeActivations;
			activationPlanes := cfg.activationPlanes;
			groupSize := cfg.groupSize; scaleMode := cfg.scaleMode;

			NEW(Gate, TritPacker.NewEmpty(hidden, emb, cfg.ffnGatePlanes, groupSize, scaleMode), FALSE);
			NEW(Up, TritPacker.NewEmpty(hidden, emb, cfg.ffnUpPlanes, groupSize, scaleMode), FALSE);
			NEW(Down, TritPacker.NewEmpty(emb, hidden, cfg.ffnDownPlanes, groupSize, scaleMode), FALSE)
		END Init;

		PROCEDURE Forward* (input: TritIO.Floats; seq: SIZE): TritIO.Floats;
		VAR gate, up, down: TritIO.Floats; i: SIZE;
		BEGIN
			gate := Gate.ForwardSequence(input, seq);
			up := Up.ForwardSequence(input, seq);

			TritOps.Silu(gate);
			FOR i := 0 TO LEN(gate) - 1 DO gate[i] := gate[i] * up[i] END;

			IF quantizeActivations THEN
				gate := TritOps.Requantize(gate, seq, hidden, activationPlanes, groupSize, scaleMode)
			END;

			down := Down.ForwardSequence(gate, seq);

			IF quantizeActivations THEN
				down := TritOps.Requantize(down, seq, emb, activationPlanes, groupSize, scaleMode)
			END;
			RETURN down
		END Forward;

		PROCEDURE Save* (w: Streams.Writer);
		BEGIN Gate.Save(w); Up.Save(w); Down.Save(w)
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN RETURN Gate.Load(r, message) & Up.Load(r, message) & Down.Load(r, message)
		END Load;

	END SwiGLU;

END TritFFN.
