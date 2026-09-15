MODULE TritPacker;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Multi-plane ternary weights: w = sum of scale[p] * t[p], t in {-1,0,+1}";
	LIMITATIONS
		"- five trits to a byte in base 3, which is 1.6 bits each rather than the 2 a
		   bit pair would spend -- that packing is what puts a whole model in kilobytes;
		 - a plane is two bitmasks, one for +1 and one for -1, so a trit-against-trit
		   dot product is two population counts per 64 weights."
*)

IMPORT Streams, Files, TritIO;

CONST
	TritsPerWord* = 64;

TYPE
	Planes = POINTER TO ARRAY OF TritIO.Words;

	Packer* = OBJECT
	VAR
		rows-, cols-, planes-: SIZE;
		groupSize-: SIZE;
		scaleMode-: SIGNED32;

		P-, N-: Planes;			(** the +1 and -1 bitmasks, [plane][row*WordsPerRow + word] *)
		scales-: TritIO.Floats;	(** laid out by scaleMode: see GetScale *)

		PROCEDURE &Init* (rows, cols, planes: SIZE);
		VAR totalWords, p: SIZE;
		BEGIN
			ASSERT((rows > 0) & (cols > 0) & (planes > 0));
			SELF.rows := rows; SELF.cols := cols; SELF.planes := planes;
			groupSize := cols; scaleMode := 0;		(* TritConfig.ScaleGlobal *)
			NEW(scales, planes);

			totalWords := rows * WordsPerRow();
			NEW(P, planes); NEW(N, planes);
			FOR p := 0 TO planes - 1 DO
				NEW(P[p], totalWords); NEW(N[p], totalWords)
			END
		END Init;

		PROCEDURE WordsPerRow* (): SIZE;
		BEGIN RETURN WordsNeeded(cols)
		END WordsPerRow;

		PROCEDURE Length* (): SIZE;
		BEGIN RETURN rows * cols
		END Length;

		PROCEDURE GroupsPerRow* (): SIZE;
		VAR g: SIZE;
		BEGIN
			g := groupSize;
			IF g <= 0 THEN g := cols END;
			RETURN (cols + g - 1) DIV g
		END GroupsPerRow;

		PROCEDURE Set* (plane, row, col: SIZE; trit: SIGNED8);
		VAR word, bit: SIZE; ps, ns: SET64;
		BEGIN
			word := row * WordsPerRow() + col DIV TritsPerWord;
			bit := col MOD TritsPerWord;
			ps := SET64(P[plane][word]); ns := SET64(N[plane][word]);
			IF trit = 1 THEN INCL(ps, bit); EXCL(ns, bit)
			ELSIF trit = -1 THEN EXCL(ps, bit); INCL(ns, bit)
			ELSE EXCL(ps, bit); EXCL(ns, bit)
			END;
			P[plane][word] := UNSIGNED64(ps); N[plane][word] := UNSIGNED64(ns)
		END Set;

		PROCEDURE Get* (plane, row, col: SIZE): SIGNED8;
		VAR word, bit: SIZE; plus, minus: BOOLEAN;
		BEGIN
			word := row * WordsPerRow() + col DIV TritsPerWord;
			bit := col MOD TritsPerWord;
			plus := bit IN SET64(P[plane][word]);
			minus := bit IN SET64(N[plane][word]);
			IF plus & ~minus THEN RETURN 1 END;
			IF ~plus & minus THEN RETURN -1 END;
			RETURN 0
		END Get;

		PROCEDURE SetFlat* (plane, index: SIZE; trit: SIGNED8);
		BEGIN Set(plane, index DIV cols, index MOD cols, trit)
		END SetFlat;

		PROCEDURE ClearPlane* (plane: SIZE);
		VAR i: SIZE;
		BEGIN
			FOR i := 0 TO LEN(P[plane]) - 1 DO P[plane][i] := 0; N[plane][i] := 0 END
		END ClearPlane;

		PROCEDURE ScaleCount* (): SIZE;
		BEGIN
			IF scaleMode = 0 THEN RETURN planes
			ELSIF scaleMode = 1 THEN RETURN rows * planes
			ELSE RETURN rows * GroupsPerRow() * planes
			END
		END ScaleCount;

		PROCEDURE GetScale* (row, group, plane: SIZE): FLOAT32;
		VAR groups: SIZE;
		BEGIN
			IF group < 0 THEN group := 0 END;
			groups := GroupsPerRow();
			IF group >= groups THEN group := groups - 1 END;

			IF scaleMode = 0 THEN RETURN scales[plane]
			ELSIF scaleMode = 1 THEN RETURN scales[row * planes + plane]
			ELSE RETURN scales[(row * groups + group) * planes + plane]
			END
		END GetScale;

		PROCEDURE SetScale (row, group, plane: SIZE; value: FLOAT32);
		VAR groups: SIZE;
		BEGIN
			IF group < 0 THEN group := 0 END;
			groups := GroupsPerRow();
			IF group >= groups THEN group := groups - 1 END;

			IF scaleMode = 0 THEN scales[plane] := value
			ELSIF scaleMode = 1 THEN scales[row * planes + plane] := value
			ELSE scales[(row * groups + group) * planes + plane] := value
			END
		END SetScale;

		(** Quantize fits each plane to what the planes before it left over. *)
		PROCEDURE Quantize (data: TritIO.Floats; scaleMode: SIGNED32; groupSize: SIZE; thresholdFraction: FLOAT32);
		VAR residual: TritIO.Floats; plane, row, col, group, groups, start, startCol, startFlat, length, c, i, flat: SIZE;
			scale, threshold: FLOAT32; t: SIGNED8;
		BEGIN
			IF groupSize <= 0 THEN SELF.groupSize := cols ELSE SELF.groupSize := groupSize END;
			SELF.scaleMode := scaleMode;
			NEW(scales, ScaleCount());

			NEW(residual, LEN(data));
			FOR i := 0 TO LEN(data) - 1 DO residual[i] := data[i] END;

			FOR plane := 0 TO planes - 1 DO
				IF scaleMode = 0 THEN							(* one scale for the whole matrix *)
					scale := MeanAbs(residual, 0, LEN(residual));
					scales[plane] := scale;
					threshold := scale * thresholdFraction;
					FOR i := 0 TO LEN(residual) - 1 DO
						t := TernaryValue(residual[i], threshold);
						SetFlat(plane, i, t);
						residual[i] := residual[i] - scale * t
					END
				ELSIF scaleMode = 1 THEN						(* one scale per row *)
					FOR row := 0 TO rows - 1 DO
						start := row * cols;
						scale := MeanAbs(residual, start, cols);
						SetScale(row, 0, plane, scale);
						threshold := scale * thresholdFraction;
						FOR col := 0 TO cols - 1 DO
							flat := start + col;
							t := TernaryValue(residual[flat], threshold);
							Set(plane, row, col, t);
							residual[flat] := residual[flat] - scale * t
						END
					END
				ELSE											(* one scale per group within a row *)
					groups := GroupsPerRow();
					FOR row := 0 TO rows - 1 DO
						FOR group := 0 TO groups - 1 DO
							startCol := group * SELF.groupSize;
							length := SELF.groupSize;
							IF startCol + length > cols THEN length := cols - startCol END;
							startFlat := row * cols + startCol;

							scale := MeanAbs(residual, startFlat, length);
							SetScale(row, group, plane, scale);
							threshold := scale * thresholdFraction;

							FOR c := 0 TO length - 1 DO
								flat := startFlat + c;
								t := TernaryValue(residual[flat], threshold);
								Set(plane, row, startCol + c, t);
								residual[flat] := residual[flat] - scale * t
							END
						END
					END
				END
			END
		END Quantize;

		(** DotRowPlane is the trit-against-trit dot product of two rows. *)
		PROCEDURE DotRowPlane* (rowA, planeA: SIZE; other: Packer; rowB, planeB: SIZE): SIZE;
		VAR words, baseA, baseB, w, plus, minus: SIZE; same, diff: SET64;
			aP, aN, bP, bN: TritIO.Words;
		BEGIN
			ASSERT(cols = other.cols);
			words := WordsNeeded(cols);
			baseA := rowA * WordsPerRow(); baseB := rowB * other.WordsPerRow();
			aP := P[planeA]; aN := N[planeA];
			bP := other.P[planeB]; bN := other.N[planeB];

			plus := 0; minus := 0;
			FOR w := 0 TO words - 1 DO
				same := (SET64(aP[baseA + w]) * SET64(bP[baseB + w]))
					+ (SET64(aN[baseA + w]) * SET64(bN[baseB + w]));
				diff := (SET64(aP[baseA + w]) * SET64(bN[baseB + w]))
					+ (SET64(aN[baseA + w]) * SET64(bP[baseB + w]));
				INC(plus, PopCount(UNSIGNED64(same)));
				INC(minus, PopCount(UNSIGNED64(diff)))
			END;
			RETURN plus - minus
		END DotRowPlane;

		(** DotRowFloat is y[o] = W[o,:] . x, the inner loop of every linear layer. *)
		PROCEDURE DotRowFloat* (row: SIZE; CONST x: ARRAY OF FLOAT32; xOffset, colStart, length: SIZE): FLOAT32;
		VAR total, acc, scale: FLOAT32; plane, offset, col, group, chunk, c: SIZE; t: SIGNED8;
		BEGIN
			IF length < 0 THEN length := cols - colStart END;
			IF length <= 0 THEN RETURN 0 END;
			ASSERT((colStart >= 0) & (colStart + length <= cols));

			(* whole row and one scale per row: skip the group bookkeeping *)
			IF (colStart = 0) & (length = cols) & (scaleMode # 2) THEN
				total := 0;
				FOR plane := 0 TO planes - 1 DO
					scale := GetScale(row, 0, plane);
					IF scale # 0 THEN
						acc := 0;
						FOR col := 0 TO cols - 1 DO
							t := Get(plane, row, col);
							IF t # 0 THEN acc := acc + t * x[xOffset + col] END
						END;
						total := total + scale * acc
					END
				END;
				RETURN total
			END;

			total := 0; offset := 0;
			WHILE offset < length DO
				col := colStart + offset;
				group := 0; chunk := length - offset;
				IF groupSize > 0 THEN
					group := col DIV groupSize;
					IF groupSize - col MOD groupSize < chunk THEN chunk := groupSize - col MOD groupSize END
				END;

				FOR plane := 0 TO planes - 1 DO
					scale := GetScale(row, group, plane);
					IF scale # 0 THEN
						acc := 0;
						FOR c := 0 TO chunk - 1 DO
							t := Get(plane, row, col + c);
							IF t # 0 THEN acc := acc + t * x[xOffset + col + c] END
						END;
						total := total + scale * acc
					END
				END;
				INC(offset, chunk)
			END;
			RETURN total
		END DotRowFloat;

		PROCEDURE DotRow* (row: SIZE; CONST x: ARRAY OF FLOAT32; xOffset: SIZE): FLOAT32;
		BEGIN RETURN DotRowFloat(row, x, xOffset, 0, -1)
		END DotRow;

		(** ToFloat rebuilds the dense matrix: the training loop quantizes through it. *)
		PROCEDURE ToFloat* (): TritIO.Floats;
		VAR result: TritIO.Floats; row, col, group, plane: SIZE; value, scale: FLOAT32;
		BEGIN
			NEW(result, Length());
			FOR row := 0 TO rows - 1 DO
				FOR col := 0 TO cols - 1 DO
					group := 0;
					IF groupSize > 0 THEN group := col DIV groupSize END;
					value := 0;
					FOR plane := 0 TO planes - 1 DO
						scale := GetScale(row, group, plane);
						IF scale # 0 THEN value := value + scale * Get(plane, row, col) END
					END;
					result[row * cols + col] := value
				END
			END;
			RETURN result
		END ToFloat;

		PROCEDURE Save* (w: Streams.Writer);
		VAR i, plane, total, chunks: SIZE; packed: TritIO.Bytes;
		BEGIN
			TritIO.WriteString(w, "MPTP_V2");

			w.RawInt32(SIGNED32(rows)); w.RawInt32(SIGNED32(cols)); w.RawInt32(SIGNED32(planes));
			w.RawInt32(SIGNED32(groupSize)); w.RawInt32(scaleMode);

			w.RawInt32(SIGNED32(LEN(scales)));
			FOR i := 0 TO LEN(scales) - 1 DO w.RawUInt16(TritIO.F32ToF16(scales[i])) END;

			total := rows * cols;
			chunks := (total + 4) DIV 5;
			FOR plane := 0 TO planes - 1 DO
				packed := EncodePlaneBase3(plane);
				ASSERT(LEN(packed) = chunks);
				w.RawInt32(SIGNED32(LEN(packed)));
				FOR i := 0 TO LEN(packed) - 1 DO w.Char(packed[i]) END
			END
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		VAR marker: ARRAY 32 OF CHAR; i, plane, totalWords, chunks, n: SIZE; v: SIGNED32;
			bits: UNSIGNED16; packed: TritIO.Bytes;

			PROCEDURE Int (): SIZE;
			VAR x: SIGNED32;
			BEGIN r.RawInt32(x); RETURN x
			END Int;

		BEGIN
			COPY("", message);
			IF ~TritIO.ReadString(r, marker) OR (marker # "MPTP_V2") THEN
				COPY("bad packer format, expected MPTP_V2", message); RETURN FALSE
			END;

			rows := Int(); cols := Int(); planes := Int(); groupSize := Int();
			r.RawInt32(scaleMode);

			n := Int();
			IF n # ScaleCount() THEN COPY("wrong number of scales", message); RETURN FALSE END;
			NEW(scales, n);
			FOR i := 0 TO n - 1 DO r.RawUInt16(bits); scales[i] := TritIO.F16ToF32(bits) END;

			chunks := (rows * cols + 4) DIV 5;
			totalWords := rows * WordsPerRow();
			NEW(P, planes); NEW(N, planes);
			FOR plane := 0 TO planes - 1 DO
				NEW(P[plane], totalWords); NEW(N[plane], totalWords)
			END;

			FOR plane := 0 TO planes - 1 DO
				n := Int();
				IF n # chunks THEN COPY("wrong base3 payload size", message); RETURN FALSE END;
				NEW(packed, n);
				FOR i := 0 TO n - 1 DO packed[i] := r.Get() END;
				DecodePlaneBase3(plane, packed)
			END;
			RETURN r.res = Streams.Ok
		END Load;

		(* EncodePlaneBase3 puts five trits in a byte: 3^5 = 243 fits in 256. *)
		PROCEDURE EncodePlaneBase3 (plane: SIZE): TritIO.Bytes;
		VAR out: TritIO.Bytes; total, chunks, index, ch, k, value, multiplier, digit: SIZE;
		BEGIN
			total := rows * cols;
			chunks := (total + 4) DIV 5;
			NEW(out, chunks);

			index := 0;
			FOR ch := 0 TO chunks - 1 DO
				value := 0; multiplier := 1; k := 0;
				WHILE (k < 5) & (index < total) DO
					CASE Get(plane, index DIV cols, index MOD cols) OF
						  -1: digit := 0
						| 1: digit := 2
					ELSE digit := 1
					END;
					INC(value, digit * multiplier);
					multiplier := multiplier * 3;
					INC(k); INC(index)
				END;
				out[ch] := CHR(value)
			END;
			RETURN out
		END EncodePlaneBase3;

		PROCEDURE DecodePlaneBase3 (plane: SIZE; data: TritIO.Bytes);
		VAR total, index, i, k, value, digit: SIZE;
		BEGIN
			total := rows * cols;
			index := 0;
			ClearPlane(plane);

			FOR i := 0 TO LEN(data) - 1 DO
				value := ORD(data[i]); k := 0;
				WHILE (k < 5) & (index < total) DO
					digit := value MOD 3;
					value := value DIV 3;
					Set(plane, index DIV cols, index MOD cols, SIGNED8(digit - 1));
					INC(k); INC(index)
				END
			END
		END DecodePlaneBase3;

	END Packer;

PROCEDURE WordsNeeded* (tritCount: SIZE): SIZE;
BEGIN RETURN (tritCount + TritsPerWord - 1) DIV TritsPerWord
END WordsNeeded;

(** FromFloat quantizes dense FP32 data into a fresh packer. *)
PROCEDURE FromFloat* (data: TritIO.Floats; rows, cols, planes, groupSize: SIZE;
	scaleMode: SIGNED32; thresholdFraction: FLOAT32): Packer;
VAR p: Packer;
BEGIN
	ASSERT(LEN(data) = rows * cols);
	NEW(p, rows, cols, planes);
	p.Quantize(data, scaleMode, groupSize, thresholdFraction);
	RETURN p
END FromFloat;

(** NewEmpty is FromFloat over zeros, without building the zero matrix twice. *)
PROCEDURE NewEmpty* (rows, cols, planes, groupSize: SIZE; scaleMode: SIGNED32): Packer;
VAR data: TritIO.Floats;
BEGIN
	NEW(data, rows * cols);
	RETURN FromFloat(data, rows, cols, planes, groupSize, scaleMode, 0)
END NewEmpty;

(* MeanAbs divides by every element, zeros included: that is the threshold the
   article's quantizer is calibrated against. *)
PROCEDURE MeanAbs (data: TritIO.Floats; start, length: SIZE): FLOAT32;
VAR sum: FLOAT32; i: SIZE;
BEGIN
	IF length <= 0 THEN RETURN 0 END;
	sum := 0;
	FOR i := 0 TO length - 1 DO sum := sum + ABS(data[start + i]) END;
	RETURN sum / length
END MeanAbs;

PROCEDURE TernaryValue (value, threshold: FLOAT32): SIGNED8;
BEGIN
	IF value > threshold THEN RETURN 1 END;
	IF value < -threshold THEN RETURN -1 END;
	RETURN 0
END TernaryValue;

(* ponytail: SWAR population count, because the machine's POPCNT is not reachable
   from Active Oberon. Only the trit-against-trit dot product uses it, and inference
   does not: an intrinsic is the upgrade path if that ever changes. *)
PROCEDURE PopCount (x: UNSIGNED64): SIZE;
CONST m1 = 5555555555555555H; m2 = 3333333333333333H; m4 = 0F0F0F0F0F0F0F0FH;
VAR n: UNSIGNED64;
BEGIN
	n := UNSIGNED64(SET64(x) * SET64(m1)) + UNSIGNED64(SET64(x DIV 2) * SET64(m1));
	n := UNSIGNED64(SET64(n) * SET64(m2)) + UNSIGNED64(SET64(n DIV 4) * SET64(m2));
	n := UNSIGNED64(SET64(n + n DIV 16) * SET64(m4));
	RETURN SIZE((n * 0101010101010101H) DIV 100000000000000H MOD 100H)
END PopCount;

PROCEDURE {TEST} TestPopCount*;
BEGIN
	ASSERT(PopCount(0) = 0);
	ASSERT(PopCount(1) = 1);
	ASSERT(PopCount(255) = 8);
	ASSERT(PopCount(0FFFFFFFFFFFFFFFFH) = 64);
	ASSERT(PopCount(0AAAAAAAAAAAAAAAAH) = 32)
END TestPopCount;

PROCEDURE {TEST} TestRoundTripAndDot*;
VAR data, x, dense: TritIO.Floats; p: Packer; row, col, i: SIZE; want, got: FLOAT32;
CONST rows = 3; cols = 70; planes = 2;		(* 70 columns crosses the 64-trit word boundary *)
BEGIN
	NEW(data, rows * cols);
	FOR i := 0 TO LEN(data) - 1 DO data[i] := (i MOD 13) - 6 END;

	p := FromFloat(data, rows, cols, planes, 32, 2, 0);
	dense := p.ToFloat();

	NEW(x, cols);
	FOR i := 0 TO cols - 1 DO x[i] := (i MOD 7) - 3 END;

	FOR row := 0 TO rows - 1 DO
		want := 0;
		FOR col := 0 TO cols - 1 DO want := want + dense[row * cols + col] * x[col] END;
		got := p.DotRow(row, x^, 0);
		ASSERT(ABS(got - want) <= 0.001 * ABS(want) + 0.001)
	END
END TestRoundTripAndDot;

PROCEDURE {TEST} TestSaveLoad*;
VAR data: TritIO.Floats; p, q: Packer; f: Files.File; w: Files.Writer; r: Files.Reader;
	i, plane, row, col: SIZE; res: INTEGER; message: ARRAY 64 OF CHAR;
CONST rows = 3; cols = 70; planes = 2;
BEGIN
	NEW(data, rows * cols);
	FOR i := 0 TO LEN(data) - 1 DO data[i] := (i MOD 11) - 5 END;
	p := FromFloat(data, rows, cols, planes, 32, 2, 0);

	f := Files.New("packer.test.bin");
	Files.OpenWriter(w, f, 0);
	p.Save(w); w.Update; Files.Register(f);

	f := Files.Old("packer.test.bin");
	ASSERT(f # NIL);
	Files.OpenReader(r, f, 0);
	NEW(q, 1, 1, 1);
	ASSERT(q.Load(r, message));

	(* not a trit may move, and the scales survive the trip through FP16 *)
	FOR plane := 0 TO planes - 1 DO
		FOR row := 0 TO rows - 1 DO
			FOR col := 0 TO cols - 1 DO
				ASSERT(p.Get(plane, row, col) = q.Get(plane, row, col))
			END
		END
	END;
	FOR i := 0 TO LEN(p.scales) - 1 DO
		ASSERT(ABS(p.scales[i] - q.scales[i]) <= 0.01 * ABS(p.scales[i]) + 0.0001)
	END;
	Files.Delete("packer.test.bin", res)
END TestSaveLoad;

END TritPacker.
