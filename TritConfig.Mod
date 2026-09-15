MODULE TritConfig;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Model geometry, written in the field order both model files share"
*)

IMPORT Streams;

CONST
	(** How many weights share one FP16 scale. *)
	ScaleGlobal* = 0; ScalePerRow* = 1; ScalePerGroup* = 2;

	(** How attention scores become weights. *)
	AttnSoftmax* = 0; AttnTopK* = 1; AttnHardmax* = 2;

TYPE
	Config* = OBJECT
	VAR
		vocabSize*, embeddingDim*: SIZE;
		hiddenDim*: SIZE;			(** below 4*emb, because SwiGLU adds a gate projection *)
		numHeads*: SIZE;
		numKVHeads*: SIZE;			(** equal to numHeads gives plain MHA, fewer gives GQA *)
		numLayers*, maxSeqLen*: SIZE;
		ropeTheta*: FLOAT32;

		embeddingPlanes*, attentionQkvPlanes*, attentionOutPlanes*: SIZE;
		ffnGatePlanes*, ffnUpPlanes*, ffnDownPlanes*: SIZE;
		lmHeadPlanes*, activationPlanes*: SIZE;

		groupSize*: SIZE;
		scaleMode*: SIGNED32;

		attentionMode*: SIGNED32;
		topK*: SIZE;

		quantizeActivations*, tiedOutput*: BOOLEAN;

		PROCEDURE &Init*;
		BEGIN
			vocabSize := 16000; embeddingDim := 128; hiddenDim := 256;
			numHeads := 4; numKVHeads := 2; numLayers := 2;
			maxSeqLen := 512; ropeTheta := 10000.0;
			SetUniformPlanes(2);
			groupSize := 32; scaleMode := ScalePerGroup;
			attentionMode := AttnSoftmax; topK := 8;
			quantizeActivations := FALSE; tiedOutput := TRUE
		END Init;

		PROCEDURE HeadDim* (): SIZE;
		BEGIN
			IF numHeads > 0 THEN RETURN embeddingDim DIV numHeads END;
			RETURN 0
		END HeadDim;

		PROCEDURE KVDim* (): SIZE;
		BEGIN RETURN numKVHeads * HeadDim()
		END KVDim;

		PROCEDURE SetUniformPlanes* (planes: SIZE);
		BEGIN
			embeddingPlanes := planes; attentionQkvPlanes := planes; attentionOutPlanes := planes;
			ffnGatePlanes := planes; ffnUpPlanes := planes; ffnDownPlanes := planes;
			lmHeadPlanes := planes; activationPlanes := planes
		END SetUniformPlanes;

		(** Validate returns an empty string when the geometry is consistent. *)
		PROCEDURE Validate* (VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN
			COPY("", message);
			IF vocabSize <= 0 THEN COPY("VocabSize must be > 0", message)
			ELSIF embeddingDim <= 0 THEN COPY("EmbeddingDim must be > 0", message)
			ELSIF hiddenDim <= 0 THEN COPY("HiddenDim must be > 0", message)
			ELSIF numHeads <= 0 THEN COPY("NumHeads must be > 0", message)
			ELSIF numKVHeads <= 0 THEN COPY("NumKVHeads must be > 0", message)
			ELSIF numKVHeads > numHeads THEN COPY("NumKVHeads must be <= NumHeads", message)
			ELSIF numHeads MOD numKVHeads # 0 THEN COPY("NumHeads must be divisible by NumKVHeads for GQA", message)
			ELSIF embeddingDim MOD numHeads # 0 THEN COPY("EmbeddingDim must be divisible by NumHeads", message)
			ELSIF HeadDim() MOD 2 # 0 THEN COPY("HeadDim must be even for RoPE", message)
			ELSIF numLayers < 0 THEN COPY("NumLayers must be >= 0", message)
			ELSIF maxSeqLen <= 0 THEN COPY("MaxSeqLen must be > 0", message)
			END;
			RETURN message = ""
		END Validate;

		(** TotalParameters counts latent FP32 weights, tied output head included as one table. *)
		PROCEDURE TotalParameters* (): SIGNED64;
		VAR vocab, emb, hidden, kvDim, perLayer, lmHead: SIGNED64;
		BEGIN
			vocab := vocabSize; emb := embeddingDim; hidden := hiddenDim; kvDim := KVDim();
			perLayer := emb * emb			(* Q *)
				+ kvDim * emb				(* K *)
				+ kvDim * emb				(* V *)
				+ emb * emb					(* attention out *)
				+ hidden * emb				(* FFN gate *)
				+ hidden * emb				(* FFN up *)
				+ emb * hidden;				(* FFN down *)
			IF tiedOutput THEN lmHead := 0 ELSE lmHead := vocab * emb END;
			RETURN vocab * emb + numLayers * perLayer + lmHead
		END TotalParameters;

	END Config;

PROCEDURE Write* (w: Streams.Writer; c: Config);
BEGIN
	w.RawInt32(SIGNED32(c.vocabSize)); w.RawInt32(SIGNED32(c.embeddingDim));
	w.RawInt32(SIGNED32(c.hiddenDim)); w.RawInt32(SIGNED32(c.numHeads));
	w.RawInt32(SIGNED32(c.numKVHeads)); w.RawInt32(SIGNED32(c.numLayers));
	w.RawInt32(SIGNED32(c.maxSeqLen)); w.RawFloat32(c.ropeTheta);

	w.RawInt32(SIGNED32(c.embeddingPlanes)); w.RawInt32(SIGNED32(c.attentionQkvPlanes));
	w.RawInt32(SIGNED32(c.attentionOutPlanes));
	w.RawInt32(SIGNED32(c.ffnGatePlanes)); w.RawInt32(SIGNED32(c.ffnUpPlanes));
	w.RawInt32(SIGNED32(c.ffnDownPlanes));
	w.RawInt32(SIGNED32(c.lmHeadPlanes)); w.RawInt32(SIGNED32(c.activationPlanes));

	w.RawInt32(SIGNED32(c.groupSize)); w.RawInt32(c.scaleMode);
	w.RawInt32(c.attentionMode); w.RawInt32(SIGNED32(c.topK));

	w.RawBool(c.quantizeActivations); w.RawBool(c.tiedOutput)
END Write;

PROCEDURE Read* (r: Streams.Reader): Config;
VAR c: Config; v: SIGNED32;

	PROCEDURE Int (): SIZE;
	VAR x: SIGNED32;
	BEGIN r.RawInt32(x); RETURN x
	END Int;

BEGIN
	NEW(c);
	c.vocabSize := Int(); c.embeddingDim := Int(); c.hiddenDim := Int();
	c.numHeads := Int(); c.numKVHeads := Int(); c.numLayers := Int();
	c.maxSeqLen := Int(); r.RawFloat32(c.ropeTheta);

	c.embeddingPlanes := Int(); c.attentionQkvPlanes := Int(); c.attentionOutPlanes := Int();
	c.ffnGatePlanes := Int(); c.ffnUpPlanes := Int(); c.ffnDownPlanes := Int();
	c.lmHeadPlanes := Int(); c.activationPlanes := Int();

	c.groupSize := Int(); r.RawInt32(c.scaleMode);
	r.RawInt32(c.attentionMode); c.topK := Int();

	r.RawBool(c.quantizeActivations); r.RawBool(c.tiedOutput);
	RETURN c
END Read;

PROCEDURE {TEST} TestGeometry*;
VAR c: Config; message: ARRAY 128 OF CHAR;
BEGIN
	NEW(c);
	ASSERT(c.Validate(message));
	ASSERT(c.HeadDim() = 32);
	ASSERT(c.KVDim() = 64);
	c.SetUniformPlanes(1);
	ASSERT(c.ffnDownPlanes = 1);

	c.numKVHeads := 3;			(* 4 heads are not divisible by 3 *)
	ASSERT(~c.Validate(message));
	ASSERT(message # "")
END TestGeometry;

PROCEDURE {TEST} TestParameterCount*;
VAR c: Config;
BEGIN
	(* the article's minimal configuration: 106 496 latent weights *)
	NEW(c);
	c.vocabSize := 1216; c.embeddingDim := 64; c.hiddenDim := 64;
	c.numHeads := 1; c.numKVHeads := 1; c.numLayers := 1;
	ASSERT(c.TotalParameters() = 1216 * 64 + 4 * 64 * 64 + 3 * 64 * 64)
END TestParameterCount;

END TritConfig.
