MODULE TritTrainer;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Turns a flat token stream into examples and runs the epoch loop"
*)

IMPORT Streams, Math, Random, TritIO, TritTensor, TritGrad, TritAdam, TritTrainModel;

TYPE
	Example = POINTER TO RECORD
		ids: TritIO.Ints
	END;
	ExampleArray = POINTER TO ARRAY OF Example;

	Trainer* = OBJECT
	VAR
		model-: TritTrainModel.Model;
		optimizer-: TritAdam.Optimizer;
		labelSmoothing*: FLOAT32;
		epoch*: SIZE;

		rng: Random.Sequence;

		PROCEDURE &Init* (model: TritTrainModel.Model; optimizer: TritAdam.Optimizer);
		BEGIN
			SELF.model := model; SELF.optimizer := optimizer;
			model.RegisterWith(optimizer);
			labelSmoothing := 0; epoch := 0;
			NEW(rng); rng.InitSeed(20260915)
		END Init;

		(** Train cuts examples at BOS..EOS boundaries, falling back to fixed windows
			when the stream carries no markers. *)
		PROCEDURE Train* (tokens: TritIO.Ints; count, epochs, batchSize: SIZE;
			log: Streams.Writer; startEpoch: SIZE);
		VAR examples, shuffled: ExampleArray; nofExamples, maxSeq, bos, eos: SIZE;
			i, j, e, b, last, baseEpoch, epochBatches, take: SIZE;
			epochLoss, avgLoss, ppl: FLOAT32;

			PROCEDURE Push (from, to: SIZE);
			VAR x: Example; bigger: ExampleArray; n: SIZE;
			BEGIN
				IF to - from < 2 THEN RETURN END;
				IF nofExamples = LEN(examples) THEN
					NEW(bigger, nofExamples * 2);
					FOR n := 0 TO nofExamples - 1 DO bigger[n] := examples[n] END;
					examples := bigger
				END;
				NEW(x); NEW(x.ids, to - from);
				FOR n := 0 TO to - from - 1 DO x.ids[n] := tokens[from + n] END;
				examples[nofExamples] := x; INC(nofExamples)
			END Push;

		BEGIN
			maxSeq := model.config.maxSeqLen;
			baseEpoch := epoch;
			IF startEpoch >= 0 THEN baseEpoch := startEpoch END;
			bos := model.tokenizer.BosId(); eos := model.tokenizer.EosId();

			NEW(examples, 64); nofExamples := 0;
			FOR i := 0 TO count - 1 DO
				IF tokens[i] = bos THEN
					last := count;
					IF i + maxSeq < last THEN last := i + maxSeq END;
					j := i + 1;
					WHILE j < last DO
						IF tokens[j] = eos THEN last := j + 1; j := last ELSE INC(j) END
					END;
					Push(i, last)
				END
			END;

			IF nofExamples = 0 THEN
				i := 0;
				WHILE i + maxSeq <= count DO Push(i, i + maxSeq); INC(i) END
			END;

			IF nofExamples = 0 THEN
				IF log # NIL THEN log.String("  No examples for training."); log.Ln; log.Update END;
				RETURN
			END;

			IF log # NIL THEN
				log.Ln;
				log.String("  Examples: "); log.Int(nofExamples, 0); log.Ln;
				log.String("  Batch size: "); log.Int(batchSize, 0); log.Ln;
				log.String("  Epochs: "); log.Int(epochs, 0); log.Ln;
				log.String("  LR: "); log.Float(optimizer.learningRate, 10); log.Ln;
				log.String("  Weight decay: "); log.Float(optimizer.weightDecay, 10); log.Ln;
				log.Ln; log.Update
			END;

			NEW(shuffled, nofExamples);
			FOR e := 0 TO epochs - 1 DO
				FOR i := 0 TO nofExamples - 1 DO shuffled[i] := examples[i] END;
				FOR i := nofExamples - 1 TO 1 BY -1 DO
					j := rng.Dice(SIGNED32(i + 1));
					Swap(shuffled, i, j)
				END;

				epochLoss := 0; epochBatches := 0;
				b := 0;
				WHILE b < nofExamples DO
					take := batchSize;
					IF b + take > nofExamples THEN take := nofExamples - b END;
					epochLoss := epochLoss + TrainBatch(shuffled, b, take);
					INC(epochBatches);
					INC(b, batchSize)
				END;

				IF (log # NIL) & ((e MOD 5 = 0) OR (e = epochs - 1)) THEN
					avgLoss := epochLoss / FLOAT32(Max(1, epochBatches));
					IF avgLoss > 20.0 THEN ppl := Math.exp(20.0) ELSE ppl := Math.exp(avgLoss) END;
					log.String("[Epoch "); log.Int(baseEpoch + e + 1, 3);
					log.String("/"); log.Int(baseEpoch + epochs, 0);
					log.String("] Loss="); log.FloatFix(avgLoss, 0, 4, 0);
					log.String(" PPL="); log.FloatFix(ppl, 0, 2, 0); log.Ln; log.Update
				END;

				epoch := baseEpoch + e + 1
			END
		END Train;

		PROCEDURE TrainBatch* (batch: ExampleArray; from, count: SIZE): FLOAT32;
		VAR total: FLOAT32; b, rows, vocab, row, v, tid: SIZE;
			input, targets: TritIO.Ints; target: TritIO.Floats;
			logits, loss: TritTensor.Node; smooth: FLOAT32;
		BEGIN
			model.ZeroGrads;
			vocab := model.config.vocabSize;
			total := 0;

			FOR b := from TO from + count - 1 DO
				rows := LEN(batch[b].ids) - 1;
				input := Slice(batch[b].ids, 0, rows);
				targets := Slice(batch[b].ids, 1, rows);

				logits := model.ForwardAllPositions(input, rows);

				NEW(target, rows * vocab);
				smooth := labelSmoothing / FLOAT32(vocab);
				FOR row := 0 TO rows - 1 DO
					tid := targets[row];
					IF (tid >= 0) & (tid < vocab) THEN
						FOR v := 0 TO vocab - 1 DO target[row * vocab + v] := smooth END;
						target[row * vocab + tid] := 1 - labelSmoothing + smooth
					END
				END;

				loss := TritGrad.CrossEntropy(logits, target);
				total := total + loss.data[0];
				TritTensor.Run(loss)
			END;

			optimizer.Step;
			RETURN total / FLOAT32(Max(1, count))
		END TrainBatch;

	END Trainer;

PROCEDURE Slice (ids: TritIO.Ints; from, count: SIZE): TritIO.Ints;
VAR out: TritIO.Ints; i: SIZE;
BEGIN
	NEW(out, count);
	FOR i := 0 TO count - 1 DO out[i] := ids[from + i] END;
	RETURN out
END Slice;

PROCEDURE Swap (list: ExampleArray; i, j: SIZE);
VAR t: Example;
BEGIN t := list[i]; list[i] := list[j]; list[j] := t
END Swap;

PROCEDURE Max (a, b: SIZE): SIZE;
BEGIN IF a > b THEN RETURN a END; RETURN b
END Max;

END TritTrainer.
