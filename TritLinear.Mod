MODULE TritLinear;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Packed linear and embedding layers: a trit row dotted with an FP32 vector"
*)

IMPORT Streams, TritIO, TritPacker;

TYPE
	Linear* = OBJECT
	VAR
		weight-: TritPacker.Packer;
		bias-: TritIO.Floats;

		PROCEDURE &Init* (weight: TritPacker.Packer; useBias: BOOLEAN);
		BEGIN
			SELF.weight := weight;
			IF useBias THEN NEW(bias, weight.rows) ELSE NEW(bias, 0) END
		END Init;

		PROCEDURE SetWeight* (w: TritPacker.Packer);
		BEGIN weight := w
		END SetWeight;

		PROCEDURE OutFeatures* (): SIZE; BEGIN RETURN weight.rows END OutFeatures;
		PROCEDURE InFeatures* (): SIZE; BEGIN RETURN weight.cols END InFeatures;

		PROCEDURE Forward* (input: TritIO.Floats): TritIO.Floats;
		BEGIN RETURN ForwardSequence(input, 1)
		END Forward;

		PROCEDURE ForwardSequence* (input: TritIO.Floats; rows: SIZE): TritIO.Floats;
		VAR out: TritIO.Floats; inF, outF, r, o: SIZE; b: FLOAT32;
		BEGIN
			inF := InFeatures(); outF := OutFeatures();
			ASSERT(LEN(input) >= rows * inF);
			NEW(out, rows * outF);
			FOR r := 0 TO rows - 1 DO
				FOR o := 0 TO outF - 1 DO
					IF o < LEN(bias) THEN b := bias[o] ELSE b := 0 END;
					out[r * outF + o] := b + weight.DotRow(o, input^, r * inF)
				END
			END;
			RETURN out
		END ForwardSequence;

		PROCEDURE Save* (w: Streams.Writer);
		BEGIN weight.Save(w); TritIO.WriteFloats(w, bias)
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN
			IF ~weight.Load(r, message) THEN RETURN FALSE END;
			bias := TritIO.ReadFloats(r);
			RETURN TRUE
		END Load;

	END Linear;

	(** Embedding is the token table, packed like any other matrix so that a tied
		output head is just a dot against one of its rows. *)
	Embedding* = OBJECT
	VAR
		table-: TritPacker.Packer;

		PROCEDURE &Init* (table: TritPacker.Packer);
		BEGIN SELF.table := table
		END Init;

		PROCEDURE SetTable* (t: TritPacker.Packer);
		BEGIN table := t
		END SetTable;

		PROCEDURE VocabSize* (): SIZE; BEGIN RETURN table.rows END VocabSize;
		PROCEDURE EmbeddingDim* (): SIZE; BEGIN RETURN table.cols END EmbeddingDim;

		PROCEDURE GetRow* (tokenId: SIZE; dst: TritIO.Floats; offset: SIZE);
		VAR dim, i, group, plane: SIZE; value: FLOAT32;
		BEGIN
			dim := EmbeddingDim();
			IF (tokenId < 0) OR (tokenId >= VocabSize()) THEN
				FOR i := 0 TO dim - 1 DO dst[offset + i] := 0 END;
				RETURN
			END;
			FOR i := 0 TO dim - 1 DO
				group := 0;
				IF table.groupSize > 0 THEN group := i DIV table.groupSize END;
				value := 0;
				FOR plane := 0 TO table.planes - 1 DO
					value := value + table.GetScale(tokenId, group, plane) * table.Get(plane, tokenId, i)
				END;
				dst[offset + i] := value
			END
		END GetRow;

		PROCEDURE ForwardSequence* (tokens: TritIO.Ints; count: SIZE): TritIO.Floats;
		VAR out: TritIO.Floats; dim, i: SIZE;
		BEGIN
			dim := EmbeddingDim();
			NEW(out, count * dim);
			FOR i := 0 TO count - 1 DO GetRow(tokens[i], out, i * dim) END;
			RETURN out
		END ForwardSequence;

		(** DotWithRow is the tied-output logit for one token. *)
		PROCEDURE DotWithRow* (tokenId: SIZE; x: TritIO.Floats; offset: SIZE): FLOAT32;
		BEGIN
			IF (tokenId < 0) OR (tokenId >= VocabSize()) THEN RETURN MIN(FLOAT32) END;
			RETURN table.DotRow(tokenId, x^, offset)
		END DotWithRow;

		PROCEDURE Save* (w: Streams.Writer);
		BEGIN table.Save(w)
		END Save;

		PROCEDURE Load* (r: Streams.Reader; VAR message: ARRAY OF CHAR): BOOLEAN;
		BEGIN RETURN table.Load(r, message)
		END Load;

	END Embedding;

PROCEDURE {TEST} TestLinearAgainstDense*;
VAR data, x, dense, out: TritIO.Floats; l: Linear; p: TritPacker.Packer;
	i, o, c: SIZE; want: FLOAT32;
CONST rows = 5; cols = 12;
BEGIN
	NEW(data, rows * cols);
	FOR i := 0 TO LEN(data) - 1 DO data[i] := (i MOD 9) - 4 END;
	p := TritPacker.FromFloat(data, rows, cols, 2, 4, 2, 0);
	dense := p.ToFloat();

	NEW(l, p, FALSE);
	NEW(x, cols);
	FOR i := 0 TO cols - 1 DO x[i] := (i MOD 5) - 2 END;

	out := l.ForwardSequence(x, 1);
	FOR o := 0 TO rows - 1 DO
		want := 0;
		FOR c := 0 TO cols - 1 DO want := want + dense[o * cols + c] * x[c] END;
		ASSERT(ABS(out[o] - want) <= 0.001 * ABS(want) + 0.001)
	END
END TestLinearAgainstDense;

PROCEDURE {TEST} TestEmbeddingRowMatchesDot*;
VAR data, row, dense: TritIO.Floats; e: Embedding; p: TritPacker.Packer; i, d: SIZE;
	dot, want: FLOAT32; tokens: TritIO.Ints;
CONST vocab = 7; dim = 8;
BEGIN
	NEW(data, vocab * dim);
	FOR i := 0 TO LEN(data) - 1 DO data[i] := (i MOD 6) - 3 END;
	p := TritPacker.FromFloat(data, vocab, dim, 1, 4, 2, 0);
	NEW(e, p);
	dense := p.ToFloat();

	NEW(row, dim);
	e.GetRow(3, row, 0);
	FOR d := 0 TO dim - 1 DO ASSERT(row[d] = dense[3 * dim + d]) END;

	(* GetRow and DotWithRow must agree: the tied head uses the second one *)
	dot := e.DotWithRow(3, row, 0);
	want := 0;
	FOR d := 0 TO dim - 1 DO want := want + dense[3 * dim + d] * row[d] END;
	ASSERT(ABS(dot - want) <= 0.001 * ABS(want) + 0.001);

	(* a token out of range embeds as zeros rather than trapping *)
	NEW(tokens, 1); tokens[0] := 999;
	row := e.ForwardSequence(tokens, 1);
	FOR d := 0 TO dim - 1 DO ASSERT(row[d] = 0) END
END TestEmbeddingRowMatchesDot;

END TritLinear.
