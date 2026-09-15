MODULE TritTrain;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "The full pipeline: create, pretrain, tune, save, export, chat";
	LIMITATIONS
		"data files are read relative to the working directory; the C# original reads
		 them from beside its executable."
*)

IMPORT Commands, Streams, Strings, Files, StdIO, Kernel,
	TritIO, TritMap, TritConfig, TritBPE, TritModel, TritTrainModel, TritTrainer,
	TritAdam, TritData;

CONST
	PretrainData = "data/pretraining_data.json";
	ChatData = "data/chat_training_data.json";
	MasterModel = "models/master_model.bin";
	TernaryModel = "models/ternary_model.bin";

VAR
	out: Streams.Writer;
	in: Streams.Reader;

PROCEDURE Do*;
VAR context: Commands.Context; master: TritTrainModel.Model; choice: ARRAY 64 OF CHAR;
BEGIN
	context := StdIO.env;
	out := context.out; in := context.in;

	master := NIL;
	WHILE master = NIL DO
		out.String("╔════════════════════════════════╗"); out.Ln;
		out.String("║   Ternary LLM  — Главное меню  ║"); out.Ln;
		out.String("╠════════════════════════════════╣"); out.Ln;
		out.String("║  1. Загрузить модель           ║"); out.Ln;
		out.String("║  2. Создать модель             ║"); out.Ln;
		out.String("╚════════════════════════════════╝"); out.Ln;
		out.Ln; out.String("Выберите опцию (1 или 2): "); out.Update;

		Line(choice);
		IF choice = "1" THEN master := LoadMaster()
		ELSIF choice = "2" THEN master := CreateMaster()
		ELSIF in.res # Streams.Ok THEN RETURN
		END
	END;

	LOOP
		out.Ln;
		out.String("╔═══════════════════════════════╗"); out.Ln;
		out.String("║          Меню действий        ║"); out.Ln;
		out.String("╠═══════════════════════════════╣"); out.Ln;
		out.String("║  Мастер-модель:               ║"); out.Ln;
		out.String("║                               ║"); out.Ln;
		out.String("║  1. Pretrain                  ║"); out.Ln;
		out.String("║  2. Tune                      ║"); out.Ln;
		out.String("║  3. Сохранить модель          ║"); out.Ln;
		out.String("║  4. Информация о модели       ║"); out.Ln;
		out.String("║  5. Экспорт тернарной модели  ║"); out.Ln;
		out.String("║  6. Тест токенизатора         ║"); out.Ln;
		out.String("╠═══════════════════════════════╣"); out.Ln;
		out.String("║  Тернарная-модель:            ║"); out.Ln;
		out.String("║                               ║"); out.Ln;
		out.String("║  7. Тест Pretrain             ║"); out.Ln;
		out.String("║  8. Интерактивный чат         ║"); out.Ln;
		out.String("╠═══════════════════════════════╣"); out.Ln;
		out.String("║  9. Выход                     ║"); out.Ln;
		out.String("╚═══════════════════════════════╝"); out.Ln;
		out.Ln; out.String("Выберите опцию: "); out.Update;

		Line(choice);
		IF in.res # Streams.Ok THEN EXIT END;

		IF choice = "1" THEN TrainOn(master, PretrainData, "PRETRAIN", 300, FALSE)
		ELSIF choice = "2" THEN TrainOn(master, ChatData, "FINETUNE (CHAT)", 700, TRUE)
		ELSIF choice = "3" THEN SaveMaster(master)
		ELSIF choice = "4" THEN ShowInfo(master)
		ELSIF choice = "5" THEN ExportTernary(master)
		ELSIF choice = "6" THEN TestTokenizer(master.tokenizer)
		ELSIF choice = "7" THEN TestPretrain(master)
		ELSIF choice = "8" THEN Chat(master)
		ELSIF choice = "9" THEN out.String("Выход из программы."); out.Ln; out.Update; EXIT
		ELSE out.String("Неверная опция."); out.Ln; out.Update
		END
	END
END Do;

PROCEDURE CreateMaster (): TritTrainModel.Model;
VAR cfg: TritConfig.Config; tokenizer: TritBPE.Tokenizer; corpus: TritMap.List;
	embDim, hidDim, heads, kvHeads, layers, maxSeq, vocabSize, planes, actual, i: SIZE;
	model: TritTrainModel.Model; message: ARRAY 128 OF CHAR;
BEGIN
	embDim := ReadInt("Embedding dim", 64, 32, 1024);
	hidDim := ReadInt("FFN hidden dim", 64, 64, 4096);
	heads := ReadInt("Num heads", 1, 1, 32);
	kvHeads := ReadInt("Num KV heads", 1, 1, heads);
	layers := ReadInt("Num layers", 1, 1, 24);
	maxSeq := ReadInt("Max seq len", 512, 16, 8192);
	vocabSize := ReadInt("BPE vocab size", 16000, 300, 32000);
	planes := ReadInt("Trit planes 1 (1.58 bit), 2 (INT3), 6 (INT6), etc", 1, 1, 100);

	corpus := PretrainTexts();
	AppendAll(corpus, ChatTexts());

	NEW(tokenizer);
	tokenizer.Train(corpus, vocabSize, 1, out);

	actual := tokenizer.VocabSize();
	IF actual # vocabSize THEN
		out.String("BPE vocab: requested "); out.Int(vocabSize, 0);
		out.String(", actual "); out.Int(actual, 0); out.String(". Using actual."); out.Ln;
		vocabSize := actual
	END;

	NEW(cfg);
	cfg.vocabSize := vocabSize; cfg.embeddingDim := embDim; cfg.hiddenDim := hidDim;
	cfg.numHeads := heads; cfg.numKVHeads := kvHeads; cfg.numLayers := layers;
	cfg.maxSeqLen := maxSeq;
	cfg.groupSize := embDim;
	IF cfg.groupSize > 64 THEN cfg.groupSize := 64 END;
	cfg.scaleMode := TritConfig.ScalePerGroup;
	cfg.attentionMode := TritConfig.AttnSoftmax;
	cfg.quantizeActivations := FALSE; cfg.tiedOutput := TRUE;
	cfg.ropeTheta := 10000.0;
	cfg.SetUniformPlanes(planes);

	IF ~cfg.Validate(message) THEN
		out.String("Ошибка конфигурации: "); out.String(message); out.Ln; out.Update;
		RETURN NIL
	END;

	NEW(model, cfg, tokenizer);
	RETURN model
END CreateMaster;

PROCEDURE LoadMaster (): TritTrainModel.Model;
VAR path: ARRAY 512 OF CHAR; model: TritTrainModel.Model; message: ARRAY 128 OF CHAR;
	f: Files.File;
BEGIN
	out.Ln; out.String("Путь к файлу модели (по умолчанию: ");
	out.String(MasterModel); out.String("): "); out.Update;
	Line(path);
	IF path = "" THEN COPY(MasterModel, path) END;

	out.String("Загрузка модели из "); out.String(path); out.String("..."); out.Ln; out.Update;
	model := TritTrainModel.Load(path, message);
	IF model = NIL THEN
		out.String("Ошибка загрузки: "); out.String(message); out.Ln; out.Update;
		RETURN NIL
	END;
	f := Files.Old(path);
	IF f # NIL THEN
		out.String(" ✓ Модель загружена ("); out.Int(Files.Length(f) DIV 1024, 0);
		out.String(" KB)"); out.Ln
	END;
	out.Update;
	RETURN model
END LoadMaster;

(* PretrainTexts and ChatTexts are the two shapes the corpus arrives in; both feed
   the tokenizer and the trainer. *)
PROCEDURE PretrainTexts (): TritMap.List;
VAR raw, list: TritMap.List; i: SIZE; clean: ARRAY 4096 OF CHAR;
BEGIN
	raw := TritData.LoadStrings(PretrainData);
	NEW(list, 16);
	FOR i := 0 TO raw.count - 1 DO
		StripEos(raw.Get(i)^, clean);
		IF clean # "" THEN list.Add(clean) END
	END;
	RETURN list
END PretrainTexts;

PROCEDURE ChatTexts (): TritMap.List;
VAR raw, list: TritMap.List; i: SIZE; formatted: ARRAY 4096 OF CHAR;
BEGIN
	raw := TritData.LoadStrings(ChatData);
	NEW(list, 16);
	FOR i := 0 TO raw.count - 1 DO
		FormatChat(raw.Get(i)^, formatted);
		IF formatted # "" THEN list.Add(formatted) END
	END;
	RETURN list
END ChatTexts;

PROCEDURE TrainOn (model: TritTrainModel.Model; CONST path, title: ARRAY OF CHAR;
	defaultEpochs: SIZE; chat: BOOLEAN);
VAR texts: TritMap.List; tokens, ids: TritIO.Ints; count, epochs, batchSize, i, j: SIZE;
	lr, weightDecay, gradientClip: FLOAT32;
	optimizer: TritAdam.Optimizer; trainer: TritTrainer.Trainer; started: SIGNED32;
BEGIN
	out.Ln; out.String("=== "); out.String(title); out.String(" ==="); out.Ln; out.Ln; out.Update;

	epochs := ReadInt("Epochs", defaultEpochs, 1, 100000);
	lr := ReadFloat("LR (0.0001..0.01)", 0.001);
	batchSize := ReadInt("Batch size", 4, 1, 64);
	weightDecay := ReadFloat("Weight decay", 0.01);
	gradientClip := ReadFloat("Gradient clip", 1.0);

	IF chat THEN texts := ChatTexts() ELSE texts := PretrainTexts() END;
	IF texts.count = 0 THEN out.String("  Нет данных."); out.Ln; out.Update; RETURN END;
	out.Ln; out.String("  Текстов: "); out.Int(texts.count, 0); out.Ln; out.Update;

	NEW(tokens, 4096); count := 0;
	FOR i := 0 TO texts.count - 1 DO
		ids := model.tokenizer.Encode(texts.Get(i)^, TRUE, TRUE);
		FOR j := 0 TO LEN(ids) - 1 DO
			IF count = LEN(tokens) THEN tokens := GrowInts(tokens) END;
			tokens[count] := ids[j]; INC(count)
		END
	END;
	out.String("  Токенов: "); out.Int(count, 0); out.Ln; out.Update;

	NEW(optimizer, lr);
	optimizer.weightDecay := weightDecay;
	optimizer.gradientClip := gradientClip;

	NEW(trainer, model, optimizer);
	trainer.epoch := model.epoch;
	trainer.labelSmoothing := model.labelSmoothing;

	started := Kernel.GetTicks();
	trainer.Train(tokens, count, epochs, batchSize, out, model.epoch);
	model.epoch := trainer.epoch;

	out.Ln; out.String("  Обучение завершено за ");
	out.FloatFix((Kernel.GetTicks() - started) / 1000.0, 0, 1, 0);
	out.String("s"); out.Ln; out.Update
END TrainOn;

PROCEDURE SaveMaster (model: TritTrainModel.Model);
VAR path: ARRAY 512 OF CHAR; f: Files.File;
BEGIN
	out.Ln; out.String("Путь для сохранения (по умолчанию: ");
	out.String(MasterModel); out.String("): "); out.Update;
	Line(path);
	IF path = "" THEN COPY(MasterModel, path) END;

	IF ~model.Save(path) THEN
		out.String("Ошибка сохранения."); out.Ln; out.Update; RETURN
	END;
	out.String(" ✓ Модель сохранена: "); out.String(path); out.Ln;
	f := Files.Old(path);
	IF f # NIL THEN
		out.String("  Размер: "); out.Int(Files.Length(f) DIV 1024, 0); out.String(" KB"); out.Ln
	END;
	out.Update
END SaveMaster;

PROCEDURE ShowInfo (model: TritTrainModel.Model);
VAR cfg: TritConfig.Config;
BEGIN
	cfg := model.config;
	out.Ln; out.String(" ИНФОРМАЦИЯ О МОДЕЛИ"); out.Ln; out.Ln;
	out.String(" Архитектура:  TernaryLlama / multi-plane ternary"); out.Ln;
	Field(" Параметры", model.TotalParameters()); out.String(" latent weights"); out.Ln;
	Field(" Vocab size", cfg.vocabSize); out.Ln;
	Field(" Embedding", cfg.embeddingDim); out.Ln;
	Field(" FFN hidden", cfg.hiddenDim); out.Ln;
	Field(" Num heads", cfg.numHeads); out.Ln;
	Field(" Num KV heads", cfg.numKVHeads); out.Ln;
	Field(" Head dim", cfg.HeadDim()); out.Ln;
	Field(" KV dim", cfg.KVDim()); out.Ln;
	Field(" Num layers", cfg.numLayers); out.Ln;
	Field(" Max seq len", cfg.maxSeqLen); out.Ln;
	out.String(" RoPE theta: "); out.FloatFix(cfg.ropeTheta, 0, 1, 0); out.Ln;

	out.Ln; out.String(" ПЛОТНОСТЬ MULTI-PLANE TERNARY"); out.Ln; out.Ln;
	Field(" Embedding planes", cfg.embeddingPlanes); out.Ln;
	Field(" Attention QKV planes", cfg.attentionQkvPlanes); out.Ln;
	Field(" Attention Out planes", cfg.attentionOutPlanes); out.Ln;
	Field(" FFN Gate planes", cfg.ffnGatePlanes); out.Ln;
	Field(" FFN Up planes", cfg.ffnUpPlanes); out.Ln;
	Field(" FFN Down planes", cfg.ffnDownPlanes); out.Ln;
	IF cfg.tiedOutput THEN out.String(" LM head       : tied with embedding"); out.Ln
	ELSE Field(" LM head planes", cfg.lmHeadPlanes); out.Ln
	END;
	IF cfg.quantizeActivations THEN Field(" Activation planes", cfg.activationPlanes); out.Ln
	ELSE out.String(" Activations   : FP32"); out.Ln
	END;
	out.Ln; out.Update
END ShowInfo;

PROCEDURE ExportTernary (model: TritTrainModel.Model);
VAR path: ARRAY 512 OF CHAR; target: TritModel.Model; f: Files.File;
BEGIN
	out.Ln; out.String("Путь к файлу модели (по умолчанию: ");
	out.String(TernaryModel); out.String("): "); out.Update;
	Line(path);
	IF path = "" THEN COPY(TernaryModel, path) END;

	target := ToTernary(model);
	IF ~target.Save(path) THEN
		out.String("Ошибка сохранения."); out.Ln; out.Update; RETURN
	END;
	f := Files.Old(path);
	out.String(" ✓ Тернарная модель сохранена: "); out.String(path);
	IF f # NIL THEN
		out.String(" ("); out.FloatFix(Files.Length(f) / 1024.0, 0, 1, 0); out.String(" KB)")
	END;
	out.Ln; out.Update
END ExportTernary;

PROCEDURE ToTernary (model: TritTrainModel.Model): TritModel.Model;
VAR target: TritModel.Model;
BEGIN
	NEW(target, model.config, model.tokenizer);
	model.SyncToInferenceModel(target);
	out.String("[Sync] FP32 -> multi-plane ternary Llama: OK"); out.Ln; out.Update;
	RETURN target
END ToTernary;

PROCEDURE TestTokenizer (tokenizer: TritBPE.Tokenizer);
VAR input: ARRAY 4096 OF CHAR; decoded: ARRAY 4096 OF CHAR; ids: TritIO.Ints; i: SIZE;
BEGIN
	out.Ln; out.String("═══ Тест BPE токенизатора ═══"); out.Ln;
	out.String("Введите текст (пустая строка = выход):"); out.Ln; out.Update;

	LOOP
		out.Ln; out.String("> "); out.Update;
		Line(input);
		IF (input = "") OR (in.res # Streams.Ok) THEN EXIT END;

		ids := tokenizer.Encode(input, FALSE, FALSE);
		out.String("  Token IDs ["); out.Int(LEN(ids), 0); out.String("]: ");
		FOR i := 0 TO LEN(ids) - 1 DO
			IF i > 0 THEN out.String(", ") END;
			out.Int(ids[i], 0)
		END;
		out.Ln;

		out.String("  Подслова: [");
		FOR i := 0 TO LEN(ids) - 1 DO
			IF i > 0 THEN out.String("|") END;
			IF (ids[i] >= 0) & (ids[i] < tokenizer.words.count) THEN
				out.String(tokenizer.words.Get(ids[i])^)
			ELSE out.String("?")
			END
		END;
		out.String("]"); out.Ln;

		tokenizer.Decode(ids, LEN(ids), decoded);
		out.String("  Decoded: """); out.String(decoded); out.String(""""); out.Ln; out.Update
	END
END TestTokenizer;

PROCEDURE TestPretrain (model: TritTrainModel.Model);
VAR target: TritModel.Model; raw: TritMap.List; i, at: SIZE;
	first, answer: ARRAY 4096 OF CHAR;
BEGIN
	target := ToTernary(model);
	raw := TritData.LoadStrings(PretrainData);
	FOR i := 0 TO raw.count - 1 DO
		COPY(raw.Get(i)^, first);
		at := Strings.Pos(" ", first);
		IF at >= 0 THEN first[at] := 0X END;
		IF target.PredictRaw(first, 0.0, answer) THEN
			out.String(first); out.String(" -> "); out.String(answer); out.String(".")
		ELSE
			out.String(first); out.String(" -> ошибка")
		END;
		out.Ln; out.Update
	END
END TestPretrain;

PROCEDURE Chat (model: TritTrainModel.Model);
VAR target: TritModel.Model; input, answer: ARRAY 4096 OF CHAR;
	temperature, value: FLOAT32; started: SIGNED32;
BEGIN
	target := ToTernary(model);
	temperature := 0.7;

	out.Ln; out.String("╔══════════════════════════════════════╗"); out.Ln;
	out.String("║         Интерактивный чат            ║"); out.Ln;
	out.String("╠══════════════════════════════════════╣"); out.Ln;
	out.String("║  exit   — выход                      ║"); out.Ln;
	out.String("║  temp N — установить temperature     ║"); out.Ln;
	out.String("╚══════════════════════════════════════╝"); out.Ln;

	LOOP
		out.Ln; out.String("[temp="); out.FloatFix(temperature, 0, 1, 0);
		out.String("] You: "); out.Update;
		Line(input);
		IF in.res # Streams.Ok THEN EXIT END;
		IF input = "exit" THEN out.String("Выход из чата."); out.Ln; out.Update; EXIT END;

		IF input # "" THEN
			IF Strings.Pos("temp ", input) = 0 THEN
				Strings.Delete(input, 0, 5);
				Strings.Trim(input, " ");
				IF ParseFloat(input, value) THEN
					temperature := ClampTemp(value);
					out.String("  Temperature установлена: ");
					out.FloatFix(temperature, 0, 2, 0); out.Ln; out.Update
				END
			ELSE
				started := Kernel.GetTicks();
				IF target.Predict(input, temperature, answer) THEN
					out.String("Assistant: "); out.String(answer); out.Ln;
					out.String("  ["); out.Int(Kernel.GetTicks() - started, 0);
					out.String(" ms]"); out.Ln
				ELSE
					out.String("  Ошибка генерации"); out.Ln
				END;
				out.Update
			END
		END
	END
END Chat;

(* FormatChat rewrites "User: ... Assistant: ..." into the dialogue markers the
   model is tuned on; anything else passes through as plain text. *)
PROCEDURE FormatChat (CONST text: ARRAY OF CHAR; VAR result: ARRAY OF CHAR);
VAR clean, userText, assistantText: ARRAY 4096 OF CHAR; userAt, assistantAt: SIZE;
BEGIN
	StripEos(text, clean);
	userAt := Strings.Pos("User:", clean);
	assistantAt := Strings.Pos("Assistant:", clean);

	IF (userAt >= 0) & (assistantAt > userAt) THEN
		Substring(clean, userAt + 5, assistantAt, userText);
		Strings.Trim(userText, " "); TrimTrailing(userText, ":");
		Substring(clean, assistantAt + 10, Strings.Length(clean), assistantText);
		Strings.Trim(assistantText, " "); TrimTrailing(assistantText, ":");

		IF (userText # "") & (assistantText # "") THEN
			COPY(TritBPE.UserToken, result);
			Strings.Append(result, " "); Strings.Append(result, userText);
			Strings.Append(result, TritBPE.AssistantToken);
			Strings.Append(result, " "); Strings.Append(result, assistantText);
			RETURN
		END
	END;
	COPY(clean, result)
END FormatChat;

PROCEDURE StripEos (CONST text: ARRAY OF CHAR; VAR result: ARRAY OF CHAR);
VAR at: SIZE;
BEGIN
	COPY(text, result);
	at := Strings.Pos("</s>", result);
	WHILE at >= 0 DO
		Strings.Delete(result, at, 4);
		at := Strings.Pos("</s>", result)
	END;
	Strings.Trim(result, " ")
END StripEos;

PROCEDURE Substring (CONST src: ARRAY OF CHAR; from, to: SIZE; VAR dst: ARRAY OF CHAR);
VAR i: SIZE;
BEGIN
	i := 0;
	WHILE (from + i < to) & (src[from + i] # 0X) DO dst[i] := src[from + i]; INC(i) END;
	dst[i] := 0X
END Substring;

PROCEDURE TrimTrailing (VAR s: ARRAY OF CHAR; CONST ch: ARRAY OF CHAR);
VAR n: SIZE;
BEGIN
	n := Strings.Length(s);
	WHILE (n > 0) & (s[n - 1] = ch[0]) DO DEC(n); s[n] := 0X END
END TrimTrailing;

PROCEDURE AppendAll (dst, src: TritMap.List);
VAR i: SIZE;
BEGIN
	FOR i := 0 TO src.count - 1 DO dst.Add(src.Get(i)^) END
END AppendAll;

PROCEDURE GrowInts (old: TritIO.Ints): TritIO.Ints;
VAR bigger: TritIO.Ints; i: SIZE;
BEGIN
	NEW(bigger, LEN(old) * 2);
	FOR i := 0 TO LEN(old) - 1 DO bigger[i] := old[i] END;
	RETURN bigger
END GrowInts;

PROCEDURE Field (CONST name: ARRAY OF CHAR; value: SIGNED64);
BEGIN out.String(name); out.String(": "); out.Int(value, 0)
END Field;

PROCEDURE Line (VAR s: ARRAY OF CHAR);
BEGIN
	in.Ln(s);
	Strings.Trim(s, " ")
END Line;

PROCEDURE ReadInt (CONST prompt: ARRAY OF CHAR; default, min, max: SIZE): SIZE;
VAR input: ARRAY 64 OF CHAR; value: SIGNED32;
BEGIN
	out.String("  "); out.String(prompt); out.String(" (по умолчанию ");
	out.Int(default, 0); out.String("): "); out.Update;
	Line(input);
	IF input = "" THEN RETURN default END;
	Strings.StrToInt32(input, value);
	IF value = 0 THEN
		out.String("  Некорректный ввод, используется "); out.Int(default, 0); out.Ln;
		RETURN default
	END;
	IF value < min THEN RETURN min END;
	IF value > max THEN RETURN max END;
	RETURN value
END ReadInt;

PROCEDURE ReadFloat (CONST prompt: ARRAY OF CHAR; default: FLOAT32): FLOAT32;
VAR input: ARRAY 64 OF CHAR; value: FLOAT32;
BEGIN
	out.String("  "); out.String(prompt); out.String(" (по умолчанию ");
	out.Float(default, 10); out.String("): "); out.Update;
	Line(input);
	IF input = "" THEN RETURN default END;
	IF ~ParseFloat(input, value) THEN
		out.String("  Некорректный ввод, используется "); out.Float(default, 10); out.Ln;
		RETURN default
	END;
	RETURN value
END ReadFloat;

PROCEDURE ParseFloat (CONST s: ARRAY OF CHAR; VAR value: FLOAT32): BOOLEAN;
VAR wide: FLOAT64;
BEGIN
	IF s = "" THEN RETURN FALSE END;
	Strings.StrToFloat(s, wide);
	value := FLOAT32(wide);
	RETURN TRUE
END ParseFloat;

PROCEDURE ClampTemp (v: FLOAT32): FLOAT32;
BEGIN
	IF v < 0.01 THEN RETURN 0.01 END;
	IF v > 2.0 THEN RETURN 2.0 END;
	RETURN v
END ClampTemp;

END TritTrain.
