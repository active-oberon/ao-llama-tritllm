MODULE TritBPE;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Byte-level BPE with GPT-2 pre-tokenization and atomic special tokens";
	LIMITATIONS
		"- .NET's pre-tokenizing pattern ends in \s+(?!\S); this engine is a Thompson
		   NFA and has no lookahead, so Pretokenize runs the pattern without that branch
		   and hands the last character of a whitespace run back to the following token
		   itself. Getting it wrong does not fail, it shifts every leading space by one
		   token -- TestPretokenizeWhitespace is what holds it;
		 - \s is spelled out to match .NET rather than an engine's own idea of it,
		   because the corpus is Russian and a non-breaking space must not split a word;
		 - no BPE cache: the C# original keeps one per encoded word, and this does the
		   merges again each time. Add one when a corpus makes it matter."
*)

IMPORT Streams, Files, Strings, Regex, UTF8Strings, TritIO, TritMap;

(* UTF8Strings.DecodeChar reports success on the terminating 0X too, so every walk
   over a string here stops on cp = 0 rather than on the call returning FALSE. *)

CONST
	PadToken* = "<pad>"; UnkToken* = "<unk>"; BosToken* = "<s>"; EosToken* = "</s>";
	UserToken* = "<user>"; AssistantToken* = "<assistant>"; SepToken* = "<sep>";

	(** what .NET's \s matches: an engine's ASCII-only \s would tokenize Russian differently *)
	Whitespace* = "\t\n\v\f\r\x{85}\p{Z}";

	MaxToken* = 512;		(** longest token or pre-token piece, in bytes *)

	pairSep = 01X;			(** never appears in a GPT-2 encoded token: every byte there is >= 21H *)

TYPE
	Token* = ARRAY MaxToken OF CHAR;

	Parts = POINTER TO RECORD
		item: Strings.StringArray;
		count: SIZE
	END;
	PartsArray = POINTER TO ARRAY OF Parts;

	Indices = POINTER TO RECORD
		item: POINTER TO ARRAY OF SIZE;
		count: SIZE
	END;
	IndicesArray = POINTER TO ARRAY OF Indices;

	Tokenizer* = OBJECT
	VAR
		vocab-: TritMap.Map;			(** token -> id *)
		words-: TritMap.List;			(** id -> token *)
		mergeA-, mergeB-: TritMap.List;	(** the merge pairs, in the order they were learned *)
		specials-: TritMap.List;

		ranks: TritMap.Map;				(** "a" + sep + "b" -> merge order *)
		re: Regex.Regexp;

		PROCEDURE &Init*;
		BEGIN
			NEW(vocab, 1024); NEW(words, 1024);
			NEW(mergeA, 1024); NEW(mergeB, 1024);
			NEW(specials, 8);
			AddDefaultSpecials(specials);
			ranks := NIL; re := NIL
		END Init;

		PROCEDURE VocabSize* (): SIZE;
		BEGIN RETURN vocab.count
		END VocabSize;

		PROCEDURE Id (CONST token: ARRAY OF CHAR; fallback: SIZE): SIZE;
		VAR id: SIZE;
		BEGIN
			id := vocab.Get(token);
			IF id = TritMap.Missing THEN RETURN fallback END;
			RETURN id
		END Id;

		PROCEDURE PadId* (): SIZE; BEGIN RETURN Id(PadToken, 0) END PadId;
		PROCEDURE UnkId* (): SIZE; BEGIN RETURN Id(UnkToken, 1) END UnkId;
		PROCEDURE BosId* (): SIZE; BEGIN RETURN Id(BosToken, 2) END BosId;
		PROCEDURE EosId* (): SIZE; BEGIN RETURN Id(EosToken, 3) END EosId;
		PROCEDURE UserId* (): SIZE; BEGIN RETURN Id(UserToken, 4) END UserId;
		PROCEDURE AssistantId* (): SIZE; BEGIN RETURN Id(AssistantToken, 5) END AssistantId;

		PROCEDURE IsSpecial (CONST s: ARRAY OF CHAR): BOOLEAN;
		VAR i: SIZE;
		BEGIN
			FOR i := 0 TO specials.count - 1 DO
				IF specials.Get(i)^ = s THEN RETURN TRUE END
			END;
			RETURN FALSE
		END IsSpecial;

		PROCEDURE Pattern (VAR pattern: ARRAY OF CHAR);
		VAR i, j, longest, at: SIZE; escaped: Token; done: POINTER TO ARRAY OF BOOLEAN;
		BEGIN
			COPY("", pattern);
			(* longest special first, so <assistant> is never cut short by a shorter one *)
			NEW(done, specials.count);
			FOR i := 0 TO specials.count - 1 DO
				longest := -1; at := -1;
				FOR j := 0 TO specials.count - 1 DO
					IF ~done[j] & (Strings.Length(specials.Get(j)^) > longest) THEN
						longest := Strings.Length(specials.Get(j)^); at := j
					END
				END;
				done[at] := TRUE;
				EscapeRegex(specials.Get(at)^, escaped);
				Strings.Append(pattern, " ?"); Strings.Append(pattern, escaped);
				Strings.Append(pattern, "|")
			END;
			Strings.Append(pattern, "'s|'t|'re|'ve|'m|'ll|'d");
			Strings.Append(pattern, "| ?\p{L}+");
			Strings.Append(pattern, "| ?\p{N}+");
			Strings.Append(pattern, "| ?[^"); Strings.Append(pattern, Whitespace);
			Strings.Append(pattern, "\p{L}\p{N}]+");
			Strings.Append(pattern, "|["); Strings.Append(pattern, Whitespace);
			Strings.Append(pattern, "]+")
		END Pattern;

		PROCEDURE Matcher (): Regex.Regexp;
		VAR pattern: ARRAY 1024 OF CHAR; error: ARRAY 128 OF CHAR;
		BEGIN
			IF re = NIL THEN
				Pattern(pattern);
				re := Regex.Compile(pattern, error);
				ASSERT(re # NIL)
			END;
			RETURN re
		END Matcher;

		(** Pretokenize splits like the GPT-2 regex, including the \s+(?!\S) branch. *)
		PROCEDURE Pretokenize* (CONST text: ARRAY OF CHAR): TritMap.List;
		VAR out: TritMap.List; r: Regex.Regexp; at, pos, len, tailStart: SIZE;
			piece, carry, joined: Token;
		BEGIN
			NEW(out, 16);
			r := Matcher();
			at := 0; COPY("", carry);
			WHILE r.Find(text, at, pos, len) DO
				Extract(text, pos, len, piece);
				IF carry # "" THEN
					COPY(carry, joined); Strings.Append(joined, piece);
					COPY(joined, piece); COPY("", carry)
				END;

				(* a whitespace run of two or more gives its last character back: with a
				   space the next token takes it, otherwise it stands on its own *)
				IF AllSpace(piece) & (Strings.Length(piece) >= 2) & (pos + len < Strings.Length(text)) THEN
					tailStart := LastCharStart(piece);
					IF piece[tailStart] = " " THEN
						COPY(" ", carry)
					END;
					IF piece[tailStart] # " " THEN
						out.Add(""); SetLast(out, piece, 0, tailStart);
						out.Add(""); SetLast(out, piece, tailStart, Strings.Length(piece) - tailStart)
					ELSE
						out.Add(""); SetLast(out, piece, 0, tailStart)
					END
				ELSE
					out.Add(piece)
				END;
				at := pos + len
			END;
			IF carry # "" THEN out.Add(carry) END;
			RETURN out
		END Pretokenize;

		(** Train builds the vocabulary: specials, the 256 byte tokens, then merges. *)
		PROCEDURE Train* (corpus: TritMap.List; vocabSize, minFrequency: SIZE;
			log: Streams.Writer);
		VAR wordIndex, pairFreq, pairSlot: TritMap.Map;
			wordText: TritMap.List;
			freq, touched: POINTER TO ARRAY OF SIZE;
			splits: PartsArray; pairWords: IndicesArray;
			pieces: TritMap.List;
			i, j, k, n, idx, step, maxMerges, bestFreq, slot, nofPairs, capacity: SIZE;
			encoded, trimmed, bestA, bestB, merged, key: Token;
			line: Strings.String;

			PROCEDURE PairKey (CONST a, b: ARRAY OF CHAR; VAR key: ARRAY OF CHAR);
			BEGIN
				COPY(a, key); Strings.AppendChar(key, pairSep); Strings.Append(key, b)
			END PairKey;

			PROCEDURE Slot (CONST key: ARRAY OF CHAR): SIZE;
			VAR s: SIZE; bigger: IndicesArray; ix: Indices;
			BEGIN
				s := pairSlot.Get(key);
				IF s # TritMap.Missing THEN RETURN s END;
				IF nofPairs = capacity THEN
					capacity := capacity * 2;
					NEW(bigger, capacity);
					FOR s := 0 TO nofPairs - 1 DO bigger[s] := pairWords[s] END;
					pairWords := bigger
				END;
				NEW(ix); NEW(ix.item, 8); ix.count := 0;
				pairWords[nofPairs] := ix;
				pairSlot.Put(key, nofPairs);
				INC(nofPairs);
				RETURN nofPairs - 1
			END Slot;

			PROCEDURE Note (CONST key: ARRAY OF CHAR; word: SIZE);
			VAR ix: Indices; bigger: POINTER TO ARRAY OF SIZE; s: SIZE;
			BEGIN
				ix := pairWords[Slot(key)];
				IF ix.count = LEN(ix.item) THEN
					NEW(bigger, ix.count * 2);
					FOR s := 0 TO ix.count - 1 DO bigger[s] := ix.item[s] END;
					ix.item := bigger
				END;
				ix.item[ix.count] := word; INC(ix.count)
			END Note;

			(* Reindex takes the merge out of one word and fixes the pair counts around it *)
			PROCEDURE Reindex (word: SIZE);
			VAR p: Parts; j: SIZE; key: Token; f: SIZE;
			BEGIN
				p := splits[word]; f := freq[word];
				FOR j := 0 TO p.count - 2 DO
					PairKey(p.item[j]^, p.item[j + 1]^, key);
					IF (p.item[j]^ # bestA) OR (p.item[j + 1]^ # bestB) THEN
						IF pairFreq.Has(key) THEN
							IF pairFreq.Add(key, -f) <= 0 THEN pairFreq.Delete(key) END
						END
					END
				END;
				splits[word] := ApplyOnce(p, bestA, bestB, merged);
				p := splits[word];
				FOR j := 0 TO p.count - 2 DO
					PairKey(p.item[j]^, p.item[j + 1]^, key);
					f := pairFreq.Add(key, freq[word]);
					Note(key, word)
				END
			END Reindex;

		BEGIN
			NEW(vocab, 1024); NEW(words, 1024);
			NEW(mergeA, 1024); NEW(mergeB, 1024);
			ranks := NIL; re := NIL;

			(* specials first, in a fixed order, then the 256 byte tokens *)
			FOR i := 0 TO specials.count - 1 DO
				IF ~vocab.Has(specials.Get(i)^) THEN
					vocab.Put(specials.Get(i)^, words.count); words.Add(specials.Get(i)^)
				END
			END;
			FOR i := 0 TO 255 DO
				ByteToken(ByteOrderAt(i), encoded);
				IF ~vocab.Has(encoded) THEN vocab.Put(encoded, words.count); words.Add(encoded) END
			END;
			IF log # NIL THEN
				log.String("Базовый vocab (specials + bytes): "); log.Int(words.count, 0); log.Ln; log.Update
			END;

			(* word frequencies over the pre-tokenized corpus *)
			NEW(wordIndex, 1024); NEW(wordText, 1024);
			NEW(freq, 1024);
			FOR i := 0 TO corpus.count - 1 DO
				line := corpus.Get(i);
				IF line^ # "" THEN
					pieces := Pretokenize(line^);
					FOR j := 0 TO pieces.count - 1 DO
						TrimLeftSpace(pieces.Get(j)^, trimmed);
						IF ~IsSpecial(trimmed) THEN
							EncodeWordToBytes(pieces.Get(j)^, encoded);
							IF encoded # "" THEN
								idx := wordIndex.Get(encoded);
								IF idx = TritMap.Missing THEN
									idx := wordText.count;
									IF idx = LEN(freq) THEN freq := GrowSizes(freq, idx * 2) END;
									wordIndex.Put(encoded, idx); wordText.Add(encoded); freq[idx] := 0
								END;
								INC(freq[idx])
							END
						END
					END
				END
			END;
			IF log # NIL THEN
				log.String("Уникальных слов: "); log.Int(wordText.count, 0); log.Ln; log.Update
			END;

			(* every word starts split into single characters *)
			NEW(splits, wordText.count);
			NEW(touched, wordText.count);
			FOR i := 0 TO wordText.count - 1 DO
				splits[i] := SplitIntoChars(wordText.Get(i)^);
				touched[i] := -1
			END;

			NEW(pairFreq, 1024); NEW(pairSlot, 1024);
			capacity := 1024; NEW(pairWords, capacity); nofPairs := 0;

			FOR i := 0 TO wordText.count - 1 DO
				FOR j := 0 TO splits[i].count - 2 DO
					PairKey(splits[i].item[j]^, splits[i].item[j + 1]^, key);
					n := pairFreq.Add(key, freq[i]);
					Note(key, i)
				END
			END;

			maxMerges := vocabSize - words.count;
			step := 0;
			WHILE step < maxMerges DO
				IF ~FindBest(pairFreq, bestA, bestB, bestFreq) OR (bestFreq < minFrequency) THEN
					step := maxMerges		(* nothing left worth merging *)
				ELSE
					COPY(bestA, merged); Strings.Append(merged, bestB);
					mergeA.Add(bestA); mergeB.Add(bestB);
					IF ~vocab.Has(merged) THEN vocab.Put(merged, words.count); words.Add(merged) END;

					IF (log # NIL) & (step MOD 50 = 0) THEN
						log.String("step="); log.Int(step, 4);
						log.String(" freq="); log.Int(bestFreq, 5);
						log.String(" '"); LogDecoded(log, bestA);
						log.String("'+'"); LogDecoded(log, bestB);
						log.String("'=>'"); LogDecoded(log, merged); log.String("'"); log.Ln; log.Update
					END;

					PairKey(bestA, bestB, key);
					slot := pairSlot.Get(key);
					pairFreq.Delete(key);
					IF slot # TritMap.Missing THEN
						FOR k := 0 TO pairWords[slot].count - 1 DO
							i := pairWords[slot].item[k];
							IF touched[i] # step THEN		(* the list may name a word twice *)
								touched[i] := step;
								Reindex(i)
							END
						END
					END;
					INC(step)
				END
			END;

			IF log # NIL THEN
				log.String("Готово: "); log.Int(mergeA.count, 0);
				log.String(" слияний, vocab="); log.Int(words.count, 0); log.Ln; log.Update
			END;


		END Train;

		PROCEDURE Ranks (): TritMap.Map;
		VAR i: SIZE; key: Token;
		BEGIN
			IF ranks = NIL THEN
				NEW(ranks, 1024);
				FOR i := 0 TO mergeA.count - 1 DO
					COPY(mergeA.Get(i)^, key);
					Strings.AppendChar(key, pairSep);
					Strings.Append(key, mergeB.Get(i)^);
					ranks.Put(key, i)
				END
			END;
			RETURN ranks
		END Ranks;

		PROCEDURE ApplyMerges (CONST encoded: ARRAY OF CHAR): Parts;
		VAR p: Parts; order: TritMap.Map; i, best, bi, r: SIZE; key, joined: Token;
		BEGIN
			p := SplitIntoChars(encoded);
			order := Ranks();
			WHILE p.count > 1 DO
				best := -1; bi := -1;
				FOR i := 0 TO p.count - 2 DO
					COPY(p.item[i]^, key); Strings.AppendChar(key, pairSep);
					Strings.Append(key, p.item[i + 1]^);
					r := order.Get(key);
					IF (r # TritMap.Missing) & ((bi < 0) OR (r < best)) THEN best := r; bi := i END
				END;
				IF bi < 0 THEN RETURN p END;
				COPY(p.item[bi]^, joined); Strings.Append(joined, p.item[bi + 1]^);
				p.item[bi] := Strings.NewString(joined);
				FOR i := bi + 1 TO p.count - 2 DO p.item[i] := p.item[i + 1] END;
				DEC(p.count)
			END;
			RETURN p
		END ApplyMerges;

		PROCEDURE Encode* (CONST text: ARRAY OF CHAR; addBos, addEos: BOOLEAN): TritIO.Ints;
		VAR pieces: TritMap.List; p: Parts; result: TritIO.Ints; n, i, j, id: SIZE;
			trimmed, encoded: Token;

			PROCEDURE Emit (id: SIZE);
			BEGIN
				IF n = LEN(result) THEN result := GrowInts(result, n * 2) END;
				result[n] := SIGNED32(id); INC(n)
			END Emit;

		BEGIN
			NEW(result, 64); n := 0;
			IF addBos & vocab.Has(BosToken) THEN Emit(vocab.Get(BosToken)) END;

			pieces := Pretokenize(text);
			FOR i := 0 TO pieces.count - 1 DO
				TrimLeftSpace(pieces.Get(i)^, trimmed);
				IF IsSpecial(trimmed) THEN
					id := vocab.Get(trimmed);
					IF id = TritMap.Missing THEN id := UnkId() END;
					Emit(id)
				ELSE
					EncodeWordToBytes(pieces.Get(i)^, encoded);
					IF encoded # "" THEN
						p := ApplyMerges(encoded);
						FOR j := 0 TO p.count - 1 DO
							id := vocab.Get(p.item[j]^);
							IF id = TritMap.Missing THEN id := UnkId() END;
							Emit(id)
						END
					END
				END
			END;

			IF addEos & vocab.Has(EosToken) THEN Emit(vocab.Get(EosToken)) END;
			RETURN Shrink(result, n)
		END Encode;

		PROCEDURE Decode* (ids: TritIO.Ints; count: SIZE; VAR text: ARRAY OF CHAR);
		VAR i, j, idx: SIZE; cp: UTF8Strings.Char32; token: Strings.String; out: SIZE;
		BEGIN
			out := 0;
			FOR i := 0 TO count - 1 DO
				idx := ids[i];
				IF (idx >= 0) & (idx < words.count) THEN
					token := words.Get(idx);
					IF IsSpecial(token^) THEN
						j := 0;
						WHILE token[j] # 0X DO
							IF out < LEN(text) - 1 THEN text[out] := token[j]; INC(out) END;
							INC(j)
						END
					ELSE
						j := 0;
						WHILE UTF8Strings.DecodeChar(token^, j, cp) & (cp # 0) DO
							IF (cp < LEN(codePointToByte)) & (codePointToByte[cp] >= 0) THEN
								IF out < LEN(text) - 1 THEN
									text[out] := CHR(codePointToByte[cp]); INC(out)
								END
							END
						END
					END
				END
			END;
			text[out] := 0X
		END Decode;

		PROCEDURE FormatPretraining* (CONST text: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
		BEGIN COPY(text, out); Strings.Trim(out, " ")
		END FormatPretraining;

		(** FormatDialogue must match the template the model was tuned on. *)
		PROCEDURE FormatDialogue* (CONST user, assistant: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
		BEGIN
			COPY(UserToken, out); Strings.Append(out, " "); Strings.Append(out, user);
			Strings.Append(out, AssistantToken);
			IF assistant # "" THEN
				Strings.Append(out, " "); Strings.Append(out, assistant);
				Strings.Append(out, EosToken)
			END
		END FormatDialogue;

		(** Save writes only the merge pairs, as ids: every token is rebuildable from them. *)
		PROCEDURE Save* (w: Streams.Writer);
		VAR i, a, b: SIZE;
		BEGIN
			TritIO.WriteString(w, "BPE_COMPACT_V1");
			w.RawInt32(SIGNED32(specials.count));
			FOR i := 0 TO specials.count - 1 DO TritIO.WriteString(w, specials.Get(i)^) END;

			w.RawInt32(SIGNED32(mergeA.count));
			FOR i := 0 TO mergeA.count - 1 DO
				a := vocab.Get(mergeA.Get(i)^); b := vocab.Get(mergeB.Get(i)^);
				ASSERT((a # TritMap.Missing) & (b # TritMap.Missing));
				TritIO.WriteVarInt(w, a); TritIO.WriteVarInt(w, b)
			END
		END Save;

	END Tokenizer;

VAR
	byteToCodePoint-: ARRAY 256 OF SIGNED32;
	codePointToByte-: ARRAY 512 OF SIGNED32;
	byteOrder: ARRAY 256 OF SIGNED32;

PROCEDURE ByteOrderAt* (i: SIZE): SIZE;
BEGIN RETURN byteOrder[i]
END ByteOrderAt;

PROCEDURE ByteToken* (b: SIZE; VAR out: ARRAY OF CHAR);
VAR i: SIZE; ok: BOOLEAN;
BEGIN
	i := 0;
	ok := UTF8Strings.EncodeChar(byteToCodePoint[b], out, i);
	out[i] := 0X
END ByteToken;

(** EncodeWordToBytes is GPT-2's byte encoding: each byte becomes one printable code point. *)
PROCEDURE EncodeWordToBytes* (CONST word: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
VAR i, j: SIZE; ok: BOOLEAN;
BEGIN
	i := 0; j := 0;
	WHILE word[i] # 0X DO
		ok := UTF8Strings.EncodeChar(byteToCodePoint[ORD(word[i])], out, j);
		INC(i)
	END;
	out[j] := 0X
END EncodeWordToBytes;

PROCEDURE Gpt2Decode* (CONST s: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
VAR i, j: SIZE; cp: UTF8Strings.Char32;
BEGIN
	i := 0; j := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) DO
		IF (cp < LEN(codePointToByte)) & (codePointToByte[cp] >= 0) THEN
			out[j] := CHR(codePointToByte[cp]); INC(j)
		END
	END;
	out[j] := 0X
END Gpt2Decode;

PROCEDURE LoadTokenizer* (r: Streams.Reader; VAR message: ARRAY OF CHAR): Tokenizer;
VAR t: Tokenizer; marker, s, a, b, merged: Token; i, n, aId, bId: SIZE;
BEGIN
	COPY("", message);
	IF ~TritIO.ReadString(r, marker) OR (marker # "BPE_COMPACT_V1") THEN
		COPY("bad tokenizer format", message); RETURN NIL
	END;

	NEW(t);
	NEW(t.specials, 8);
	n := ReadInt(r);
	FOR i := 0 TO n - 1 DO
		IF ~TritIO.ReadString(r, s) THEN COPY("truncated special token", message); RETURN NIL END;
		t.specials.Add(s)
	END;

	NEW(t.vocab, 1024); NEW(t.words, 1024);
	NEW(t.mergeA, 1024); NEW(t.mergeB, 1024);

	FOR i := 0 TO t.specials.count - 1 DO
		t.vocab.Put(t.specials.Get(i)^, t.words.count); t.words.Add(t.specials.Get(i)^)
	END;
	FOR i := 0 TO 255 DO
		ByteToken(byteOrder[i], s);
		IF ~t.vocab.Has(s) THEN t.vocab.Put(s, t.words.count); t.words.Add(s) END
	END;

	n := ReadInt(r);
	FOR i := 0 TO n - 1 DO
		aId := TritIO.ReadVarInt(r); bId := TritIO.ReadVarInt(r);
		IF (aId < 0) OR (aId >= t.words.count) OR (bId < 0) OR (bId >= t.words.count) THEN
			COPY("compact BPE: a merge points at a token that does not exist", message);
			RETURN NIL
		END;
		COPY(t.words.Get(aId)^, a); COPY(t.words.Get(bId)^, b);
		COPY(a, merged); Strings.Append(merged, b);
		IF t.vocab.Has(merged) THEN
			COPY("compact BPE: duplicate merge token", message); RETURN NIL
		END;
		t.vocab.Put(merged, t.words.count); t.words.Add(merged);
		t.mergeA.Add(a); t.mergeB.Add(b)
	END;
	RETURN t
END LoadTokenizer;

PROCEDURE ReadInt (r: Streams.Reader): SIZE;
VAR x: SIGNED32;
BEGIN r.RawInt32(x); RETURN x
END ReadInt;

PROCEDURE AddDefaultSpecials (list: TritMap.List);
BEGIN
	list.Add(PadToken); list.Add(UnkToken); list.Add(BosToken); list.Add(EosToken);
	list.Add(UserToken); list.Add(AssistantToken); list.Add(SepToken)
END AddDefaultSpecials;

(* FindBest takes the most frequent pair, ties broken by the smaller merged string,
   so the answer does not depend on the order the map happens to iterate in. *)
PROCEDURE FindBest (pairFreq: TritMap.Map; VAR bestA, bestB: ARRAY OF CHAR; VAR bestFreq: SIZE): BOOLEAN;
VAR slot, value: SIZE; key: Strings.String; found: BOOLEAN;
	a, b, merged, bestMerged: Token;
BEGIN
	bestFreq := -1; found := FALSE; COPY("", bestMerged);
	slot := 0;
	WHILE pairFreq.Iterate(slot, key, value) DO
		SplitPair(key^, a, b);
		COPY(a, merged); Strings.Append(merged, b);
		IF (value > bestFreq) OR ((value = bestFreq) & (merged < bestMerged)) THEN
			COPY(a, bestA); COPY(b, bestB); COPY(merged, bestMerged);
			bestFreq := value; found := TRUE
		END
	END;
	RETURN found
END FindBest;

PROCEDURE SplitPair (CONST key: ARRAY OF CHAR; VAR a, b: ARRAY OF CHAR);
VAR i, j: SIZE;
BEGIN
	i := 0;
	WHILE (key[i] # 0X) & (key[i] # pairSep) DO a[i] := key[i]; INC(i) END;
	a[i] := 0X;
	j := 0;
	IF key[i] = pairSep THEN
		INC(i);
		WHILE key[i] # 0X DO b[j] := key[i]; INC(i); INC(j) END
	END;
	b[j] := 0X
END SplitPair;

(* SplitIntoChars counts with the same walk it splits by. UTF8Strings.Length steps
   through a lead-byte table while DecodeChar validates as it goes, so on a sequence
   that is not well-formed the two disagree -- and the array would come out short. *)
PROCEDURE SplitIntoChars (CONST s: ARRAY OF CHAR): Parts;
VAR p: Parts; i, j, n: SIZE; cp: UTF8Strings.Char32; one: ARRAY 8 OF CHAR;
BEGIN
	n := 0; i := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) DO INC(n) END;

	NEW(p);
	NEW(p.item, n + 1); p.count := 0;
	i := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) DO
		j := 0;
		IF UTF8Strings.EncodeChar(cp, one, j) THEN one[j] := 0X END;
		p.item[p.count] := Strings.NewString(one); INC(p.count)
	END;
	RETURN p
END SplitIntoChars;

PROCEDURE ApplyOnce (p: Parts; CONST a, b, merged: ARRAY OF CHAR): Parts;
VAR out: Parts; i: SIZE;
BEGIN
	NEW(out); NEW(out.item, p.count); out.count := 0;
	i := 0;
	WHILE i < p.count DO
		IF (i < p.count - 1) & (p.item[i]^ = a) & (p.item[i + 1]^ = b) THEN
			out.item[out.count] := Strings.NewString(merged); INC(out.count); INC(i, 2)
		ELSE
			out.item[out.count] := p.item[i]; INC(out.count); INC(i)
		END
	END;
	RETURN out
END ApplyOnce;

PROCEDURE Extract (CONST src: ARRAY OF CHAR; pos, len: SIZE; VAR dst: ARRAY OF CHAR);
VAR i: SIZE;
BEGIN
	FOR i := 0 TO len - 1 DO dst[i] := src[pos + i] END;
	dst[len] := 0X
END Extract;

PROCEDURE SetLast (list: TritMap.List; CONST src: ARRAY OF CHAR; pos, len: SIZE);
VAR part: Token;
BEGIN
	Extract(src, pos, len, part);
	list.Set(list.count - 1, part)
END SetLast;

PROCEDURE AllSpace (CONST s: ARRAY OF CHAR): BOOLEAN;
VAR i: SIZE; cp: UTF8Strings.Char32;
BEGIN
	i := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) DO
		IF ~IsSpaceCodePoint(cp) THEN RETURN FALSE END
	END;
	RETURN s # ""
END AllSpace;

(* IsSpaceCodePoint is .NET's char.IsWhiteSpace: the separator categories plus the
   five control characters and NEL. *)
PROCEDURE IsSpaceCodePoint (cp: UTF8Strings.Char32): BOOLEAN;
BEGIN
	RETURN ((cp >= 09H) & (cp <= 0DH)) OR (cp = 20H) OR (cp = 85H) OR (cp = 0A0H)
		OR (cp = 1680H) OR ((cp >= 2000H) & (cp <= 200AH))
		OR (cp = 2028H) OR (cp = 2029H) OR (cp = 202FH) OR (cp = 205FH) OR (cp = 3000H)
END IsSpaceCodePoint;

PROCEDURE LastCharStart (CONST s: ARRAY OF CHAR): SIZE;
VAR i, previous: SIZE; cp: UTF8Strings.Char32;
BEGIN
	i := 0; previous := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) DO
		IF s[i] # 0X THEN previous := i END
	END;
	RETURN previous
END LastCharStart;

PROCEDURE TrimLeftSpace (CONST s: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
VAR i, j, start: SIZE; cp: UTF8Strings.Char32;
BEGIN
	i := 0; start := 0;
	WHILE UTF8Strings.DecodeChar(s, i, cp) & (cp # 0) & IsSpaceCodePoint(cp) DO start := i END;
	j := 0;
	WHILE s[start + j] # 0X DO out[j] := s[start + j]; INC(j) END;
	out[j] := 0X
END TrimLeftSpace;

PROCEDURE EscapeRegex (CONST s: ARRAY OF CHAR; VAR out: ARRAY OF CHAR);
VAR i, j: SIZE;
BEGIN
	i := 0; j := 0;
	WHILE s[i] # 0X DO
		IF (s[i] = "\") OR (s[i] = ".") OR (s[i] = "+") OR (s[i] = "*") OR (s[i] = "?")
			OR (s[i] = "(") OR (s[i] = ")") OR (s[i] = "|") OR (s[i] = "[") OR (s[i] = "]")
			OR (s[i] = "{") OR (s[i] = "}") OR (s[i] = "^") OR (s[i] = "$") THEN
			out[j] := "\"; INC(j)
		END;
		out[j] := s[i]; INC(j); INC(i)
	END;
	out[j] := 0X
END EscapeRegex;

PROCEDURE LogDecoded (w: Streams.Writer; CONST s: ARRAY OF CHAR);
VAR plain: Token;
BEGIN Gpt2Decode(s, plain); w.String(plain)
END LogDecoded;

PROCEDURE GrowSizes (old: POINTER TO ARRAY OF SIZE; size: SIZE): POINTER TO ARRAY OF SIZE;
VAR bigger: POINTER TO ARRAY OF SIZE; i: SIZE;
BEGIN
	NEW(bigger, size);
	FOR i := 0 TO LEN(old) - 1 DO bigger[i] := old[i] END;
	RETURN bigger
END GrowSizes;

PROCEDURE GrowInts (old: TritIO.Ints; size: SIZE): TritIO.Ints;
VAR bigger: TritIO.Ints; i: SIZE;
BEGIN
	NEW(bigger, size);
	FOR i := 0 TO LEN(old) - 1 DO bigger[i] := old[i] END;
	RETURN bigger
END GrowInts;

PROCEDURE Shrink (old: TritIO.Ints; n: SIZE): TritIO.Ints;
VAR out: TritIO.Ints; i: SIZE;
BEGIN
	NEW(out, n);
	FOR i := 0 TO n - 1 DO out[i] := old[i] END;
	RETURN out
END Shrink;

(* BuildByteTables is GPT-2's bytes_to_unicode: the printable bytes stand for
   themselves and everything else is moved above 255, so no token holds a control
   character and the vocabulary order matches the C# original's. *)
PROCEDURE BuildByteTables;
VAR i, b, n: SIZE; seen: ARRAY 256 OF BOOLEAN; printable: ARRAY 256 OF BOOLEAN;

	PROCEDURE Add (b: SIZE);
	BEGIN
		IF ~seen[b] THEN seen[b] := TRUE; byteOrder[n] := SIGNED32(b); INC(n) END
	END Add;

BEGIN
	FOR i := 0 TO 255 DO seen[i] := FALSE; printable[i] := FALSE END;
	FOR i := 33 TO 126 DO printable[i] := TRUE END;
	FOR i := 161 TO 172 DO printable[i] := TRUE END;
	FOR i := 174 TO 255 DO printable[i] := TRUE END;

	n := 0;
	FOR i := 33 TO 126 DO Add(i) END;
	FOR i := 161 TO 172 DO Add(i) END;
	FOR i := 174 TO 255 DO Add(i) END;
	FOR i := 0 TO 255 DO Add(i) END;

	FOR i := 0 TO LEN(codePointToByte) - 1 DO codePointToByte[i] := -1 END;
	n := 0;
	FOR i := 0 TO 255 DO
		b := byteOrder[i];
		IF printable[b] THEN
			byteToCodePoint[b] := SIGNED32(b)
		ELSE
			byteToCodePoint[b] := SIGNED32(256 + n); INC(n)
		END;
		codePointToByte[byteToCodePoint[b]] := SIGNED32(b)
	END
END BuildByteTables;

PROCEDURE Expect (list: TritMap.List; i: SIZE; CONST want: ARRAY OF CHAR);
BEGIN
	ASSERT(i < list.count);
	ASSERT(list.Get(i)^ = want)
END Expect;

(* TestPretokenizeWhitespace pins the branch that stands in for .NET's \s+(?!\S):
   get it wrong and every leading space lands on the wrong token. *)
PROCEDURE {TEST} TestPretokenizeWhitespace*;
VAR t: Tokenizer; p: TritMap.List; nl: ARRAY 8 OF CHAR;
BEGIN
	NEW(t);

	p := t.Pretokenize("a b");
	ASSERT(p.count = 2); Expect(p, 0, "a"); Expect(p, 1, " b");

	p := t.Pretokenize("a  b");
	ASSERT(p.count = 3); Expect(p, 0, "a"); Expect(p, 1, " "); Expect(p, 2, " b");

	p := t.Pretokenize("a   b");
	ASSERT(p.count = 3); Expect(p, 0, "a"); Expect(p, 1, "  "); Expect(p, 2, " b");

	nl[0] := "a"; nl[1] := 0AX; nl[2] := "b"; nl[3] := 0X;
	p := t.Pretokenize(nl);
	ASSERT(p.count = 3); Expect(p, 0, "a"); Expect(p, 2, "b");

	p := t.Pretokenize("hello  ");
	ASSERT(p.count = 2); Expect(p, 0, "hello"); Expect(p, 1, "  ");

	p := t.Pretokenize("Привет, мир!");
	ASSERT(p.count = 4);
	Expect(p, 0, "Привет"); Expect(p, 1, ","); Expect(p, 2, " мир"); Expect(p, 3, "!");

	p := t.Pretokenize("it's");
	ASSERT(p.count = 2); Expect(p, 0, "it"); Expect(p, 1, "'s");

	p := t.Pretokenize("<user> привет<assistant>");
	ASSERT(p.count = 3);
	Expect(p, 0, "<user>"); Expect(p, 1, " привет"); Expect(p, 2, "<assistant>");

	p := t.Pretokenize("2 + 2");
	ASSERT(p.count = 3); Expect(p, 0, "2"); Expect(p, 1, " +"); Expect(p, 2, " 2")
END TestPretokenizeWhitespace;

PROCEDURE {TEST} TestByteTables*;
VAR i, n: SIZE;
BEGIN
	(* printable bytes stand for themselves, the rest are moved above 255 *)
	ASSERT(byteToCodePoint[ORD("A")] = ORD("A"));
	ASSERT(byteToCodePoint[32] = 288);					(* space becomes U+0120, GPT-2 s Ġ *)
	ASSERT(byteToCodePoint[0] > 255);
	n := 0;
	FOR i := 0 TO 255 DO
		ASSERT(codePointToByte[byteToCodePoint[i]] = i);
		IF byteToCodePoint[i] > 255 THEN INC(n) END
	END;
	ASSERT(n = 256 - 94 - 12 - 82)						(* the three printable ranges *)
END TestByteTables;

PROCEDURE {TEST} TestEncodeDecodeBytes*;
VAR encoded, plain: Token;
BEGIN
	EncodeWordToBytes(" привет", encoded);
	Gpt2Decode(encoded, plain);
	ASSERT(plain = " привет");
	EncodeWordToBytes("", encoded);
	ASSERT(encoded = "")
END TestEncodeDecodeBytes;

PROCEDURE {TEST} TestTrainEncodeDecode*;
VAR t: Tokenizer; corpus: TritMap.List; ids: TritIO.Ints; text: ARRAY 256 OF CHAR; i: SIZE;
BEGIN
	NEW(corpus, 8);
	corpus.Add("привет как дела");
	corpus.Add("привет мир");
	corpus.Add("как тебя зовут");
	corpus.Add("меня зовут модель");

	NEW(t);
	t.Train(corpus, 400, 1, NIL);

	(* the specials keep ids 0..6, which the file format and the sampler rely on *)
	ASSERT(t.vocab.Get(PadToken) = 0);
	ASSERT(t.vocab.Get(UnkToken) = 1);
	ASSERT(t.vocab.Get(BosToken) = 2);
	ASSERT(t.vocab.Get(EosToken) = 3);
	ASSERT(t.VocabSize() > 263);

	(* byte-level BPE is lossless, corpus or no corpus *)
	ids := t.Encode("привет как дела", FALSE, FALSE);
	t.Decode(ids, LEN(ids), text);
	ASSERT(text = "привет как дела");

	ids := t.Encode("совершенно новый текст 42", FALSE, FALSE);
	t.Decode(ids, LEN(ids), text);
	ASSERT(text = "совершенно новый текст 42");

	ids := t.Encode("привет", TRUE, TRUE);
	ASSERT(LEN(ids) >= 2);
	ASSERT(ids[0] = t.BosId());
	ASSERT(ids[LEN(ids) - 1] = t.EosId())
END TestTrainEncodeDecode;

PROCEDURE {TEST} TestTokenizerSaveLoad*;
VAR t, back: Tokenizer; corpus: TritMap.List; f: Files.File; w: Files.Writer; r: Files.Reader;
	a, b: TritIO.Ints; i, res: SIZE; message: ARRAY 64 OF CHAR; ignore: INTEGER;
BEGIN
	NEW(corpus, 4);
	corpus.Add("привет мир"); corpus.Add("привет как дела"); corpus.Add("мир большой");

	NEW(t); t.Train(corpus, 350, 1, NIL);

	f := Files.New("bpe.test.bin");
	Files.OpenWriter(w, f, 0);
	t.Save(w); w.Update; Files.Register(f);

	f := Files.Old("bpe.test.bin");
	Files.OpenReader(r, f, 0);
	back := LoadTokenizer(r, message);
	ASSERT(back # NIL);
	ASSERT(back.VocabSize() = t.VocabSize());

	a := t.Encode("привет как дела мир", FALSE, FALSE);
	b := back.Encode("привет как дела мир", FALSE, FALSE);
	ASSERT(LEN(a) = LEN(b));
	FOR i := 0 TO LEN(a) - 1 DO ASSERT(a[i] = b[i]) END;
	Files.Delete("bpe.test.bin", ignore)
END TestTokenizerSaveLoad;


BEGIN
	BuildByteTables
END TritBPE.
