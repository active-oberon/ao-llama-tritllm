MODULE TritModel;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "The deployable model: packed weights, a tokenizer, and generation";
	LIMITATIONS
		"Forward returns the logits of the last position only -- generation never
		 needs the others, and with a tied head each one costs a pass over the vocabulary."
*)

IMPORT Files, Streams, Strings, Random, Math,
	TritIO, TritConfig, TritPacker, TritBPE, TritLinear, TritBlock, TritOps;

CONST
	Magic = "TERNARY_LLAMA_V1";
	SampleTopKDefault = 40;

TYPE
	BlockArray = POINTER TO ARRAY OF TritBlock.Block;
	CacheArray* = POINTER TO ARRAY OF TritOps.KVCache;

	Model* = OBJECT
	VAR
		config-: TritConfig.Config;
		vocab, emb, maxSeq: SIZE;

		embedding-: TritLinear.Embedding;
		blocks-: BlockArray;
		finalNorm-: TritIO.Floats;
		lmHead-: TritLinear.Linear;
		tokenizer-: TritBPE.Tokenizer;

		rng: Random.Sequence;

		PROCEDURE &Init* (cfg: TritConfig.Config; tokenizer: TritBPE.Tokenizer);
		VAR i: SIZE; message: ARRAY 128 OF CHAR;
		BEGIN
			ASSERT(cfg.Validate(message));
			config := cfg; SELF.tokenizer := tokenizer;
			vocab := cfg.vocabSize; emb := cfg.embeddingDim; maxSeq := cfg.maxSeqLen;

			finalNorm := TritOps.Ones(emb);
			NEW(embedding, TritPacker.NewEmpty(vocab, emb, cfg.embeddingPlanes,
				cfg.groupSize, cfg.scaleMode));

			NEW(blocks, cfg.numLayers);
			FOR i := 0 TO cfg.numLayers - 1 DO NEW(blocks[i], cfg) END;

			IF ~cfg.tiedOutput THEN
				NEW(lmHead, TritPacker.NewEmpty(vocab, emb, cfg.lmHeadPlanes,
					cfg.groupSize, cfg.scaleMode), FALSE)
			END;

			NEW(rng); rng.InitSeed(20260915)
		END Init;

		PROCEDURE CreateCaches* (): CacheArray;
		VAR caches: CacheArray; i: SIZE;
		BEGIN
			NEW(caches, LEN(blocks));
			FOR i := 0 TO LEN(caches) - 1 DO NEW(caches[i], maxSeq, config.KVDim()) END;
			RETURN caches
		END CreateCaches;

		PROCEDURE Forward* (tokens: TritIO.Ints; count: SIZE; caches: CacheArray;
			VAR logits: TritIO.Floats): BOOLEAN;
		VAR hidden, last: TritIO.Floats; i, v, lastOffset: SIZE;
		BEGIN
			IF count = 0 THEN RETURN FALSE END;
			IF LEN(caches) # LEN(blocks) THEN RETURN FALSE END;
			IF (LEN(caches) > 0) & (caches[0].length + count > maxSeq) THEN RETURN FALSE END;

			hidden := embedding.ForwardSequence(tokens, count);
			IF config.quantizeActivations THEN
				hidden := TritOps.Requantize(hidden, count, emb, config.activationPlanes,
					config.groupSize, config.scaleMode)
			END;

			FOR i := 0 TO LEN(blocks) - 1 DO
				IF ~blocks[i].Forward(hidden, count, caches[i], hidden) THEN RETURN FALSE END
			END;

			hidden := TritOps.RmsNorm(hidden, count, emb, finalNorm);
			lastOffset := (count - 1) * emb;

			IF ~config.tiedOutput THEN
				NEW(last, emb);
				FOR i := 0 TO emb - 1 DO last[i] := hidden[lastOffset + i] END;
				logits := lmHead.Forward(last);
				RETURN TRUE
			END;

			NEW(logits, vocab);
			FOR v := 0 TO vocab - 1 DO
				logits[v] := embedding.DotWithRow(v, hidden, lastOffset)
			END;
			RETURN TRUE
		END Forward;

		PROCEDURE Predict* (CONST userInput: ARRAY OF CHAR; temperature: FLOAT32;
			VAR out: ARRAY OF CHAR): BOOLEAN;
		VAR prompt: ARRAY 4096 OF CHAR;
		BEGIN
			tokenizer.FormatDialogue(userInput, "", prompt);
			RETURN PredictRaw(prompt, temperature, out)
		END Predict;

		PROCEDURE PredictRaw* (CONST prompt: ARRAY OF CHAR; temperature: FLOAT32;
			VAR out: ARRAY OF CHAR): BOOLEAN;
		VAR tokens, generated, one: TritIO.Ints; logits: TritIO.Floats;
			caches: CacheArray; n, contextLen, maxPrompt, nextId: SIZE;
		BEGIN
			COPY("", out);
			tokens := tokenizer.Encode(prompt, TRUE, FALSE);
			IF LEN(tokens) = 0 THEN RETURN TRUE END;

			maxPrompt := maxSeq - 10;
			IF maxPrompt < 1 THEN maxPrompt := 1 END;
			IF LEN(tokens) > maxPrompt THEN tokens := Truncate(tokens, maxPrompt) END;

			caches := CreateCaches();
			IF ~Forward(tokens, LEN(tokens), caches, logits) THEN RETURN FALSE END;

			NEW(generated, maxSeq); n := 0;
			NEW(one, 1);
			contextLen := LEN(tokens);

			WHILE contextLen < maxSeq - 1 DO
				MaskLogits(logits);
				IF temperature <= 0.05 THEN nextId := ArgMax(logits)
				ELSE nextId := SampleTopK(logits, temperature, SampleTopKDefault)
				END;

				IF (nextId < 0) OR (nextId >= vocab)
					OR (nextId = tokenizer.EosId()) OR (nextId = tokenizer.PadId()) THEN
					contextLen := maxSeq				(* stop *)
				ELSE
					generated[n] := SIGNED32(nextId); INC(n);
					one[0] := SIGNED32(nextId);
					IF ~Forward(one, 1, caches, logits) THEN RETURN FALSE END;
					INC(contextLen)
				END
			END;

			tokenizer.Decode(generated, n, out);
			RETURN TRUE
		END PredictRaw;

		PROCEDURE MaskLogits (logits: TritIO.Floats);
		VAR validVocab, i, pad: SIZE;
		BEGIN
			validVocab := vocab;
			IF tokenizer.VocabSize() < validVocab THEN validVocab := tokenizer.VocabSize() END;
			FOR i := validVocab TO LEN(logits) - 1 DO logits[i] := TritOps.NegInfinity END;
			pad := tokenizer.PadId();
			IF (pad >= 0) & (pad < LEN(logits)) THEN logits[pad] := TritOps.NegInfinity END
		END MaskLogits;

		PROCEDURE SampleTopK* (logits: TritIO.Floats; temperature: FLOAT32; topK: SIZE): SIZE;
		BEGIN RETURN Sample(logits, temperature, topK, rng)
		END SampleTopK;

		(** ImportWeights packs the trained FP32 weights into this model. *)
		PROCEDURE ImportWeights* (embPacker: TritPacker.Packer; layers: TritOps.LayerWeightsArray;
			finalNormGamma, lmHeadData: TritIO.Floats);
		VAR cfg: TritConfig.Config; i, d: SIZE; b: TritBlock.Block;

			PROCEDURE Pack (data: TritIO.Floats; rows, cols, planes: SIZE): TritPacker.Packer;
			BEGIN
				RETURN TritPacker.FromFloat(data, rows, cols, planes, cfg.groupSize, cfg.scaleMode, 0)
			END Pack;

		BEGIN
			cfg := config;
			embedding.SetTable(embPacker);
			NEW(finalNorm, LEN(finalNormGamma));
			FOR d := 0 TO LEN(finalNormGamma) - 1 DO finalNorm[d] := finalNormGamma[d] END;

			FOR i := 0 TO LEN(blocks) - 1 DO
				IF i < LEN(layers) THEN
					b := blocks[i];
					FOR d := 0 TO emb - 1 DO
						b.norm1[d] := layers[i].norm1[d];
						b.norm2[d] := layers[i].norm2[d]
					END;

					b.attention.Q.SetWeight(Pack(layers[i].q, emb, emb, cfg.attentionQkvPlanes));
					b.attention.K.SetWeight(Pack(layers[i].k, cfg.KVDim(), emb, cfg.attentionQkvPlanes));
					b.attention.V.SetWeight(Pack(layers[i].v, cfg.KVDim(), emb, cfg.attentionQkvPlanes));
					b.attention.Out.SetWeight(Pack(layers[i].out, emb, emb, cfg.attentionOutPlanes));

					b.ffn.Gate.SetWeight(Pack(layers[i].gate, cfg.hiddenDim, emb, cfg.ffnGatePlanes));
					b.ffn.Up.SetWeight(Pack(layers[i].up, cfg.hiddenDim, emb, cfg.ffnUpPlanes));
					b.ffn.Down.SetWeight(Pack(layers[i].down, emb, cfg.hiddenDim, cfg.ffnDownPlanes))
				END
			END;

			IF ~cfg.tiedOutput & (lmHead # NIL) & (lmHeadData # NIL) THEN
				lmHead.SetWeight(Pack(lmHeadData, cfg.vocabSize, emb, cfg.lmHeadPlanes))
			END
		END ImportWeights;

		PROCEDURE TotalParameters* (): SIGNED64;
		BEGIN RETURN config.TotalParameters()
		END TotalParameters;

		PROCEDURE Save* (CONST path: ARRAY OF CHAR): BOOLEAN;
		VAR f: Files.File; w: Files.Writer; i: SIZE;
		BEGIN
			EnsureDirectory(path);
			f := Files.New(path);
			IF f = NIL THEN RETURN FALSE END;
			Files.OpenWriter(w, f, 0);

			TritIO.WriteString(w, Magic);
			TritConfig.Write(w, config);
			tokenizer.Save(w);

			embedding.Save(w);
			TritIO.WriteFloats(w, finalNorm);
			FOR i := 0 TO LEN(blocks) - 1 DO blocks[i].Save(w) END;
			w.RawBool(lmHead # NIL);
			IF lmHead # NIL THEN lmHead.Save(w) END;

			w.Update; Files.Register(f);
			RETURN TRUE
		END Save;

	END Model;

PROCEDURE Truncate (ids: TritIO.Ints; n: SIZE): TritIO.Ints;
VAR out: TritIO.Ints; i: SIZE;
BEGIN
	NEW(out, n);
	FOR i := 0 TO n - 1 DO out[i] := ids[i] END;
	RETURN out
END Truncate;

(** Sample draws from the k most likely tokens after temperature scaling. Module
	level and with the generator passed in, so the master model draws the same way
	without carrying a second copy of this. *)
PROCEDURE Sample* (logits: TritIO.Floats; temperature: FLOAT32; topK: SIZE;
	rng: Random.Sequence): SIZE;
VAR order: TritIO.Ints; probs: TritIO.Floats; n, k, i: SIZE;
	maxLogit, sum, r, cumulative: FLOAT32;
BEGIN
	n := LEN(logits);
	IF n = 0 THEN RETURN 0 END;
	IF temperature < 0.01 THEN temperature := 0.01 END;
	IF temperature > 2.0 THEN temperature := 2.0 END;
	k := topK;
	IF k < 1 THEN k := 1 END;
	IF k > n THEN k := n END;

	order := TopIndices(logits, k);
	maxLogit := logits[order[0]] / temperature;

	NEW(probs, k); sum := 0;
	FOR i := 0 TO k - 1 DO
		probs[i] := Math.exp(logits[order[i]] / temperature - maxLogit);
		sum := sum + probs[i]
	END;
	IF sum <= 1.0E-12 THEN RETURN order[0] END;

	r := rng.Uniform();
	cumulative := 0;
	FOR i := 0 TO k - 1 DO
		cumulative := cumulative + probs[i] / sum;
		IF r <= cumulative THEN RETURN order[i] END
	END;
	RETURN order[k - 1]
END Sample;

PROCEDURE ArgMax* (logits: TritIO.Floats): SIZE;
VAR best, i: SIZE;
BEGIN
	best := 0;
	FOR i := 1 TO LEN(logits) - 1 DO
		IF logits[i] > logits[best] THEN best := i END
	END;
	RETURN best
END ArgMax;

(* TopIndices returns the k largest by selection: k is 40 against a vocabulary of
   thousands, so picking k times beats sorting the whole thing, which is what the
   C# original did. *)
PROCEDURE TopIndices (logits: TritIO.Floats; k: SIZE): TritIO.Ints;
VAR order: TritIO.Ints; taken: POINTER TO ARRAY OF BOOLEAN; i, j, best: SIZE;
BEGIN
	NEW(order, k); NEW(taken, LEN(logits));
	FOR i := 0 TO k - 1 DO
		best := -1;
		FOR j := 0 TO LEN(logits) - 1 DO
			IF ~taken[j] & ((best < 0) OR (logits[j] > logits[best])) THEN best := j END
		END;
		taken[best] := TRUE;
		order[i] := SIGNED32(best)
	END;
	RETURN order
END TopIndices;

PROCEDURE EnsureDirectory* (CONST path: ARRAY OF CHAR);
VAR dir, name: ARRAY 512 OF CHAR; res: INTEGER;
BEGIN
	Files.SplitPath(path, dir, name);
	IF dir # "" THEN Files.CreateDirectory(dir, res) END
END EnsureDirectory;

PROCEDURE Load* (CONST path: ARRAY OF CHAR; VAR message: ARRAY OF CHAR): Model;
VAR f: Files.File; r: Files.Reader; magic: ARRAY 64 OF CHAR;
	cfg: TritConfig.Config; tokenizer: TritBPE.Tokenizer; m: Model; i: SIZE; hasLmHead: BOOLEAN;
BEGIN
	COPY("", message);
	f := Files.Old(path);
	IF f = NIL THEN COPY("file not found", message); RETURN NIL END;
	Files.OpenReader(r, f, 0);

	IF ~TritIO.ReadString(r, magic) OR (magic # Magic) THEN
		COPY("invalid file format", message); RETURN NIL
	END;

	cfg := TritConfig.Read(r);
	tokenizer := TritBPE.LoadTokenizer(r, message);
	IF tokenizer = NIL THEN RETURN NIL END;

	NEW(m, cfg, tokenizer);
	IF ~m.embedding.Load(r, message) THEN RETURN NIL END;
	m.finalNorm := TritIO.ReadFloats(r);
	FOR i := 0 TO LEN(m.blocks) - 1 DO
		IF ~m.blocks[i].Load(r, message) THEN RETURN NIL END
	END;

	r.RawBool(hasLmHead);
	IF hasLmHead THEN
		IF m.lmHead = NIL THEN
			COPY("LM head in file but the model is tied", message); RETURN NIL
		END;
		IF ~m.lmHead.Load(r, message) THEN RETURN NIL END
	END;
	RETURN m
END Load;

END TritModel.
