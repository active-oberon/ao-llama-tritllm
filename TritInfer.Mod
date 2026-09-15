MODULE TritInfer;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Chat against a packed ternary model";
	LIMITATIONS
		"--prompt takes the whole rest of the command line, because the arguments
		 reach an A2 program already joined into one string: a quoted argument and
		 several bare ones cannot be told apart afterwards."
*)

IMPORT Commands, Streams, Strings, StdIO, Kernel, TritModel;

CONST
	DefaultModel = "models/ternary_model.bin";

PROCEDURE Do*;
VAR context: Commands.Context; arg, modelPath, prompt, answer: ARRAY 4096 OF CHAR;
	message: ARRAY 128 OF CHAR;
	model: TritModel.Model; temperature, value: FLOAT32;
	hasPrompt, first: BOOLEAN;
BEGIN
	context := StdIO.env;
	COPY(DefaultModel, modelPath);
	temperature := 0.7; hasPrompt := FALSE; first := TRUE;

	IF context.arg.GetString(arg) THEN END;		(* argv[0] *)
	WHILE ~hasPrompt & context.arg.GetString(arg) DO
		IF arg = "--help" THEN Usage(context.out); RETURN
		ELSIF arg = "--temp" THEN
			IF context.arg.GetString(arg) & Number(arg, value) THEN
				temperature := Clamp(value)
			END
		ELSIF arg = "--prompt" THEN
			context.arg.Ln(prompt);
			Strings.Trim(prompt, " ");
			hasPrompt := TRUE
		ELSIF first THEN
			COPY(arg, modelPath)
		END;
		first := FALSE
	END;

	context.out.String("Загрузка тернарной модели из "); context.out.String(modelPath);
	context.out.String("..."); context.out.Ln; context.out.Update;

	model := TritModel.Load(modelPath, message);
	IF model = NIL THEN
		context.out.String("Ошибка загрузки модели: "); context.out.String(message);
		context.out.Ln; Usage(context.out); context.out.Update; RETURN
	END;
	context.out.String(" ✓ Модель загружена"); context.out.Ln;
	context.out.String("   Параметров: "); context.out.Int(model.TotalParameters(), 0);
	context.out.Ln; context.out.Ln; context.out.Update;

	IF hasPrompt THEN
		context.out.String("[temp="); context.out.FloatFix(temperature, 0, 1, 0);
		context.out.String("] You: "); context.out.String(prompt); context.out.Ln;
		Respond(context, model, prompt, temperature);
		RETURN
	END;

	context.out.String("╔═══════════════════════════════════════╗"); context.out.Ln;
	context.out.String("║  Ternary LLM — инференс (без обучения)║"); context.out.Ln;
	context.out.String("╠═══════════════════════════════════════╣"); context.out.Ln;
	context.out.String("║  exit   — выход                       ║"); context.out.Ln;
	context.out.String("║  temp N — установить temperature      ║"); context.out.Ln;
	context.out.String("╚═══════════════════════════════════════╝"); context.out.Ln;

	LOOP
		context.out.Ln;
		context.out.String("[temp="); context.out.FloatFix(temperature, 0, 1, 0);
		context.out.String("] You: "); context.out.Update;
		context.in.Ln(arg);
		IF context.in.res # Streams.Ok THEN EXIT END;
		Strings.Trim(arg, " ");
		IF arg = "exit" THEN EXIT END;
		IF arg # "" THEN
			IF (Strings.Pos("temp ", arg) = 0) THEN
				Strings.Delete(arg, 0, 5);
				Strings.Trim(arg, " ");
				IF Number(arg, value) THEN
					temperature := Clamp(value);
					context.out.String("  Temperature установлена: ");
					context.out.FloatFix(temperature, 0, 2, 0); context.out.Ln
				END
			ELSE
				Respond(context, model, arg, temperature)
			END
		END
	END
END Do;

PROCEDURE Respond (context: Commands.Context; model: TritModel.Model;
	CONST text: ARRAY OF CHAR; temperature: FLOAT32);
VAR answer: ARRAY 4096 OF CHAR; started: SIGNED32;
BEGIN
	started := Kernel.GetTicks();
	IF ~model.Predict(text, temperature, answer) THEN
		context.out.String("  Ошибка генерации"); context.out.Ln; context.out.Update; RETURN
	END;
	context.out.String("Assistant: "); context.out.String(answer); context.out.Ln;
	context.out.String("  ["); context.out.Int(Kernel.GetTicks() - started, 0);
	context.out.String(" ms]"); context.out.Ln; context.out.Update
END Respond;

PROCEDURE Number (CONST s: ARRAY OF CHAR; VAR value: FLOAT32): BOOLEAN;
VAR wide: FLOAT64;
BEGIN
	IF s = "" THEN RETURN FALSE END;
	Strings.StrToFloat(s, wide);
	value := FLOAT32(wide);
	RETURN TRUE
END Number;

PROCEDURE Clamp (v: FLOAT32): FLOAT32;
BEGIN
	IF v < 0.01 THEN RETURN 0.01 END;
	IF v > 2.0 THEN RETURN 2.0 END;
	RETURN v
END Clamp;

PROCEDURE Usage (w: Streams.Writer);
BEGIN
	w.String("Использование:"); w.Ln;
	w.String("  tritinfer [путь/к/ternary_model.bin] [--temp T] [--prompt текст]"); w.Ln;
	w.Ln;
	w.String("  Без --prompt запускается интерактивный чат."); w.Ln;
	w.String("  --prompt берёт весь остаток командной строки."); w.Ln;
	w.Update
END Usage;

END TritInfer.
