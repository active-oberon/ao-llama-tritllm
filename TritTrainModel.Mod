MODULE TritTrainModel;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "The FP32 master model: what you train, and what you export a packed model from";
	LIMITATIONS
		"PredictRaw here has no KV cache and re-runs the whole prefix per token. It is
		 for checking that training took, not for serving -- TritModel is that."
*)

IMPORT Files, Streams, Random,
	TritIO, TritConfig, TritPacker, TritBPE, TritTensor, TritGrad,
	TritTrainBlock, TritAdam, TritOps, TritModel;

CONST
	Magic = "TRAINABLE_LLAMA_WEIGHTS_V1";
	MaxParameters = 512;

TYPE
	Node = TritTensor.Node;
	NodeArray* = POINTER TO ARRAY OF Node;
	BlockArray = POINTER TO ARRAY OF TritTrainBlock.Block;

	(** EmbedNode gathers embedding rows; its backward scatters the gradient back. *)
	EmbedNode = OBJECT (TritTensor.Node)
	VAR table: Node; tokens: TritIO.Ints; seq, emb, vocab: SIZE;
		PROCEDURE Backward*;
		VAR i, d, tid: SIZE;
		BEGIN
			IF table.grad = NIL THEN RETURN END;
			FOR i := 0 TO seq - 1 DO
				tid := tokens[i];
				IF (tid >= 0) & (tid < vocab) THEN
					FOR d := 0 TO emb - 1 DO
						table.grad[tid * emb + d] := table.grad[tid * emb + d] + grad[i * emb + d]
					END
				END
			END
		END Backward;
	END EmbedNode;

	SliceLastNode = OBJECT (TritTensor.Node)
	VAR x: Node; seq, emb: SIZE;
		PROCEDURE Backward*;
		VAR d: SIZE;
		BEGIN
			IF x.grad = NIL THEN RETURN END;
			FOR d := 0 TO emb - 1 DO
				x.grad[(seq - 1) * emb + d] := x.grad[(seq - 1) * emb + d] + grad[d]
			END
		END Backward;
	END SliceLastNode;

	Model* = OBJECT
	VAR
		config-: TritConfig.Config;
		tokenEmbedding-: Node;
		blocks-: BlockArray;
		finalNormGamma-: Node;
		lmHead-: Node;
		tokenizer-: TritBPE.Tokenizer;

		epoch*: SIZE;
		labelSmoothing*: FLOAT32;

		rng: Random.Sequence;

		PROCEDURE &Init* (cfg: TritConfig.Config; tokenizer: TritBPE.Tokenizer);
		VAR i: SIZE; message: ARRAY 128 OF CHAR;
		BEGIN
			ASSERT(cfg.Validate(message));
			IF cfg.vocabSize # tokenizer.VocabSize() THEN cfg.vocabSize := tokenizer.VocabSize() END;
			config := cfg; SELF.tokenizer := tokenizer;

			tokenEmbedding := TritTrainBlock.Normal(NewRng(42), cfg.vocabSize, cfg.embeddingDim, 0.02);
			finalNormGamma := TritTensor.Ones(cfg.embeddingDim, TRUE);

			NEW(blocks, cfg.numLayers);
			FOR i := 0 TO cfg.numLayers - 1 DO
				NEW(blocks[i], cfg.embeddingDim, cfg.numHeads, cfg.numKVHeads, cfg.hiddenDim)
			END;

			IF ~cfg.tiedOutput THEN
				lmHead := TritTrainBlock.Normal(NewRng(4242), cfg.vocabSize, cfg.embeddingDim, 0.02)
			END;
			epoch := 0; labelSmoothing := 0;
			rng := NewRng(20260915)
		END Init;

		PROCEDURE Parameters* (VAR list: ARRAY OF Node): SIZE;
		VAR n, i: SIZE;
		BEGIN
			n := 0;
			list[n] := tokenEmbedding; INC(n);
			list[n] := finalNormGamma; INC(n);
			FOR i := 0 TO LEN(blocks) - 1 DO blocks[i].Parameters(list, n) END;
			IF lmHead # NIL THEN list[n] := lmHead; INC(n) END;
			RETURN n
		END Parameters;

		PROCEDURE ZeroGrads*;
		VAR list: ARRAY MaxParameters OF Node; n, i: SIZE;
		BEGIN
			n := Parameters(list);
			FOR i := 0 TO n - 1 DO list[i].ZeroGrad END
		END ZeroGrads;

		PROCEDURE RegisterWith* (optimizer: TritAdam.Optimizer);
		VAR list: ARRAY MaxParameters OF Node; n, i: SIZE;
		BEGIN
			n := Parameters(list);
			FOR i := 0 TO n - 1 DO optimizer.RegisterOne(list[i]) END
		END RegisterWith;

		PROCEDURE EmbedTokens (tokens: TritIO.Ints; count: SIZE): Node;
		VAR n: EmbedNode; packer: TritPacker.Packer; quant, xData: TritIO.Floats;
			emb, vocab, i, d, tid: SIZE;
		BEGIN
			emb := config.embeddingDim; vocab := config.vocabSize;
			packer := TritPacker.FromFloat(tokenEmbedding.data, vocab, emb,
				config.embeddingPlanes, config.groupSize, config.scaleMode, 0);
			quant := packer.ToFloat();

			NEW(xData, count * emb);
			FOR i := 0 TO count - 1 DO
				tid := tokens[i];
				IF (tid >= 0) & (tid < vocab) THEN
					FOR d := 0 TO emb - 1 DO xData[i * emb + d] := quant[tid * emb + d] END
				END
			END;

			NEW(n);
			TritTensor.Setup(n, xData, count, emb, TRUE);
			n.table := tokenEmbedding; n.tokens := tokens;
			n.seq := count; n.emb := emb; n.vocab := vocab;
			n.AddChild(tokenEmbedding);
			RETURN n
		END EmbedTokens;

		(* ComputeLogits ties the output head to the embedding table unless the
		   configuration asked for one of its own. *)
		PROCEDURE ComputeLogits (x: Node): Node;
		VAR weight: Node; planes: SIZE;
		BEGIN
			weight := tokenEmbedding; planes := config.embeddingPlanes;
			IF lmHead # NIL THEN weight := lmHead; planes := config.lmHeadPlanes END;
			RETURN TritGrad.MatMul(x, TritGrad.Transpose(TritGrad.Quantize(weight, planes, config)))
		END ComputeLogits;

		PROCEDURE Trunk (tokens: TritIO.Ints; count: SIZE): Node;
		VAR x: Node; i: SIZE;
		BEGIN
			x := EmbedTokens(tokens, count);
			FOR i := 0 TO LEN(blocks) - 1 DO x := blocks[i].Forward(x, config) END;
			RETURN TritGrad.RmsNorm(x, finalNormGamma)
		END Trunk;

		(** ForwardAllPositions returns logits for every position: training needs all. *)
		PROCEDURE ForwardAllPositions* (tokens: TritIO.Ints; count: SIZE): Node;
		BEGIN RETURN ComputeLogits(Trunk(tokens, count))
		END ForwardAllPositions;

		PROCEDURE Forward* (tokens: TritIO.Ints; count: SIZE): Node;
		BEGIN RETURN ComputeLogits(SliceLast(Trunk(tokens, count), config.embeddingDim))
		END Forward;

		(** SyncToInferenceModel packs the trained weights into the deployable model. *)
		PROCEDURE SyncToInferenceModel* (target: TritModel.Model);
		VAR embPacker: TritPacker.Packer; layers: TritOps.LayerWeightsArray;
			cfg: TritConfig.Config; i: SIZE; lmHeadData: TritIO.Floats;
		BEGIN
			cfg := target.config;
			embPacker := TritPacker.FromFloat(tokenEmbedding.data, cfg.vocabSize, cfg.embeddingDim,
				cfg.embeddingPlanes, cfg.groupSize, cfg.scaleMode, 0);

			NEW(layers, LEN(blocks));
			FOR i := 0 TO LEN(blocks) - 1 DO
				layers[i].norm1 := blocks[i].norm1Gamma.data;
				layers[i].norm2 := blocks[i].norm2Gamma.data;
				layers[i].q := blocks[i].qWeight.data;
				layers[i].k := blocks[i].kWeight.data;
				layers[i].v := blocks[i].vWeight.data;
				layers[i].out := blocks[i].outWeight.data;
				layers[i].gate := blocks[i].ffnGateWeight.data;
				layers[i].up := blocks[i].ffnUpWeight.data;
				layers[i].down := blocks[i].ffnDownWeight.data
			END;

			IF lmHead # NIL THEN lmHeadData := lmHead.data END;
			target.ImportWeights(embPacker, layers, finalNormGamma.data, lmHeadData)
		END SyncToInferenceModel;

		PROCEDURE PredictRaw* (CONST prompt: ARRAY OF CHAR; temperature: FLOAT32;
			VAR out: ARRAY OF CHAR);
		VAR tokens, context, generated: TritIO.Ints; logits: TritIO.Floats;
			n, contextLen, limit, nextId, step, i: SIZE;
		BEGIN
			COPY("", out);
			tokens := tokenizer.Encode(prompt, TRUE, FALSE);
			IF LEN(tokens) = 0 THEN RETURN END;

			limit := config.maxSeqLen - 10;
			IF limit < 1 THEN limit := 1 END;
			IF LEN(tokens) > limit THEN tokens := Head(tokens, limit) END;

			NEW(context, config.maxSeqLen);
			FOR i := 0 TO LEN(tokens) - 1 DO context[i] := tokens[i] END;
			contextLen := LEN(tokens);

			NEW(generated, config.maxSeqLen); n := 0;

			FOR step := 0 TO config.maxSeqLen - 1 DO
				IF contextLen < config.maxSeqLen - 1 THEN
					logits := Forward(context, contextLen).data;
					MaskLogits(logits);
					IF temperature <= 0.05 THEN nextId := TritModel.ArgMax(logits)
					ELSE nextId := TritModel.Sample(logits, temperature, 40, rng)
					END;

					IF (nextId < 0) OR (nextId >= config.vocabSize)
						OR (nextId = tokenizer.EosId()) OR (nextId = tokenizer.PadId()) THEN
						contextLen := config.maxSeqLen		(* stop *)
					ELSE
						generated[n] := SIGNED32(nextId); INC(n);
						context[contextLen] := SIGNED32(nextId); INC(contextLen)
					END
				END
			END;

			tokenizer.Decode(generated, n, out)
		END PredictRaw;

		PROCEDURE MaskLogits (logits: TritIO.Floats);
		VAR validVocab, i, pad: SIZE;
		BEGIN
			validVocab := config.vocabSize;
			IF tokenizer.VocabSize() < validVocab THEN validVocab := tokenizer.VocabSize() END;
			FOR i := validVocab TO LEN(logits) - 1 DO logits[i] := TritOps.NegInfinity END;
			pad := tokenizer.PadId();
			IF (pad >= 0) & (pad < LEN(logits)) THEN logits[pad] := TritOps.NegInfinity END
		END MaskLogits;

		PROCEDURE TotalParameters* (): SIGNED64;
		BEGIN RETURN config.TotalParameters()
		END TotalParameters;

		PROCEDURE Save* (CONST path: ARRAY OF CHAR): BOOLEAN;
		VAR f: Files.File; w: Files.Writer; i: SIZE;
		BEGIN
			TritModel.EnsureDirectory(path);
			f := Files.New(path);
			IF f = NIL THEN RETURN FALSE END;
			Files.OpenWriter(w, f, 0);

			TritIO.WriteString(w, Magic);
			TritConfig.Write(w, config);
			tokenizer.Save(w);

			TritIO.WriteFloats(w, tokenEmbedding.data);
			TritIO.WriteFloats(w, finalNormGamma.data);

			w.RawInt32(SIGNED32(LEN(blocks)));
			FOR i := 0 TO LEN(blocks) - 1 DO
				TritIO.WriteFloats(w, blocks[i].norm1Gamma.data);
				TritIO.WriteFloats(w, blocks[i].qWeight.data);
				TritIO.WriteFloats(w, blocks[i].kWeight.data);
				TritIO.WriteFloats(w, blocks[i].vWeight.data);
				TritIO.WriteFloats(w, blocks[i].outWeight.data);
				TritIO.WriteFloats(w, blocks[i].norm2Gamma.data);
				TritIO.WriteFloats(w, blocks[i].ffnGateWeight.data);
				TritIO.WriteFloats(w, blocks[i].ffnUpWeight.data);
				TritIO.WriteFloats(w, blocks[i].ffnDownWeight.data)
			END;

			w.RawBool(lmHead # NIL);
			IF lmHead # NIL THEN TritIO.WriteFloats(w, lmHead.data) END;

			w.Update; Files.Register(f);
			RETURN TRUE
		END Save;

	END Model;

PROCEDURE NewRng (seed: SIGNED32): Random.Sequence;
VAR r: Random.Sequence;
BEGIN NEW(r); r.InitSeed(seed); RETURN r
END NewRng;

PROCEDURE Head (ids: TritIO.Ints; n: SIZE): TritIO.Ints;
VAR out: TritIO.Ints; i: SIZE;
BEGIN
	NEW(out, n);
	FOR i := 0 TO n - 1 DO out[i] := ids[i] END;
	RETURN out
END Head;

PROCEDURE SliceLast (x: Node; emb: SIZE): Node;
VAR n: SliceLastNode; data: TritIO.Floats; d, seq: SIZE;
BEGIN
	seq := x.rows;
	NEW(data, emb);
	FOR d := 0 TO emb - 1 DO data[d] := x.data[(seq - 1) * emb + d] END;

	NEW(n);
	TritTensor.Setup(n, data, 1, emb, TRUE);
	n.x := x; n.seq := seq; n.emb := emb;
	n.AddChild(x);
	RETURN n
END SliceLast;

PROCEDURE Load* (CONST path: ARRAY OF CHAR; VAR message: ARRAY OF CHAR): Model;
VAR f: Files.File; r: Files.Reader; magic: ARRAY 64 OF CHAR;
	cfg: TritConfig.Config; tokenizer: TritBPE.Tokenizer; m: Model;
	i, blockCount: SIZE; hasLmHead: BOOLEAN;
	list: ARRAY MaxParameters OF Node; n: SIZE;

	PROCEDURE Into (target: Node; CONST name: ARRAY OF CHAR): BOOLEAN;
	VAR data: TritIO.Floats; j: SIZE;
	BEGIN
		data := TritIO.ReadFloats(r);
		IF LEN(data) # LEN(target.data) THEN
			COPY("wrong tensor size for ", message); Append(message, name);
			RETURN FALSE
		END;
		FOR j := 0 TO LEN(data) - 1 DO target.data[j] := data[j] END;
		target.ZeroGrad;
		RETURN TRUE
	END Into;

BEGIN
	COPY("", message);
	f := Files.Old(path);
	IF f = NIL THEN COPY("file not found", message); RETURN NIL END;
	Files.OpenReader(r, f, 0);

	IF ~TritIO.ReadString(r, magic) OR (magic # Magic) THEN
		COPY("invalid format", message); RETURN NIL
	END;

	cfg := TritConfig.Read(r);
	tokenizer := TritBPE.LoadTokenizer(r, message);
	IF tokenizer = NIL THEN RETURN NIL END;

	NEW(m, cfg, tokenizer);
	IF ~Into(m.tokenEmbedding, "TokenEmbedding") THEN RETURN NIL END;
	IF ~Into(m.finalNormGamma, "FinalNormGamma") THEN RETURN NIL END;

	blockCount := ReadInt(r);
	IF blockCount # LEN(m.blocks) THEN COPY("wrong block count", message); RETURN NIL END;

	FOR i := 0 TO blockCount - 1 DO
		IF ~Into(m.blocks[i].norm1Gamma, "Norm1Gamma") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].qWeight, "QWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].kWeight, "KWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].vWeight, "VWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].outWeight, "OutWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].norm2Gamma, "Norm2Gamma") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].ffnGateWeight, "FfnGateWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].ffnUpWeight, "FfnUpWeight") THEN RETURN NIL END;
		IF ~Into(m.blocks[i].ffnDownWeight, "FfnDownWeight") THEN RETURN NIL END
	END;

	r.RawBool(hasLmHead);
	IF hasLmHead # (m.lmHead # NIL) THEN
		COPY("LM head presence does not match the configuration", message); RETURN NIL
	END;
	IF hasLmHead THEN
		IF ~Into(m.lmHead, "LmHead") THEN RETURN NIL END
	END;

	n := m.Parameters(list);
	FOR i := 0 TO n - 1 DO list[i].ZeroGrad END;
	RETURN m
END Load;

PROCEDURE ReadInt (r: Streams.Reader): SIZE;
VAR x: SIGNED32;
BEGIN r.RawInt32(x); RETURN x
END ReadInt;

PROCEDURE Append (VAR dst: ARRAY OF CHAR; CONST src: ARRAY OF CHAR);
VAR i, j: SIZE;
BEGIN
	i := 0; WHILE dst[i] # 0X DO INC(i) END;
	j := 0; WHILE (src[j] # 0X) & (i < LEN(dst) - 1) DO dst[i] := src[j]; INC(i); INC(j) END;
	dst[i] := 0X
END Append;

END TritTrainModel.
