MODULE TritMap;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "String-keyed hash map and string list for the BPE tokenizer";
	LIMITATIONS
		"- the standard library has no map keyed by string content: FoxBasic.HashTable
		   hashes the pointer, and StringPool interns into one global pool that BPE's
		   throwaway pair keys would never leave. So this is the map, and nothing else
		   here needs one;
		 - open addressing with tombstones: a delete leaves a marker, and a table that
		   is deleted from far more often than inserted into will keep probing past them
		   until the next grow."
*)

IMPORT Strings;

CONST
	Missing* = -1;	(** what Get returns for a key that is not there *)

	empty = 0; used = 1; deleted = 2;

TYPE
	Entry = RECORD
		key: Strings.String;
		value: SIZE;
		state: SIGNED8
	END;
	EntryArray = POINTER TO ARRAY OF Entry;

	Map* = OBJECT
	VAR
		entries: EntryArray;
		count-: SIZE;		(** live keys *)
		occupied: SIZE;		(** live keys plus tombstones: what the load factor counts *)

		PROCEDURE &Init* (initial: SIZE);
		BEGIN
			IF initial < 8 THEN initial := 8 END;
			NEW(entries, RoundUpPowerOfTwo(initial));
			count := 0; occupied := 0
		END Init;

		(* Slot finds the key, or the first free slot to put it in. *)
		PROCEDURE Slot (CONST key: ARRAY OF CHAR): SIZE;
		VAR i, mask, free: SIZE;
		BEGIN
			mask := LEN(entries) - 1;
			i := Hash(key) MOD (mask + 1);
			free := -1;
			LOOP
				IF entries[i].state = empty THEN
					IF free >= 0 THEN RETURN free END;
					RETURN i
				ELSIF entries[i].state = deleted THEN
					IF free < 0 THEN free := i END
				ELSIF entries[i].key^ = key THEN
					RETURN i
				END;
				i := (i + 1) MOD (mask + 1)
			END
		END Slot;

		PROCEDURE Get* (CONST key: ARRAY OF CHAR): SIZE;
		VAR i: SIZE;
		BEGIN
			i := Slot(key);
			IF entries[i].state = used THEN RETURN entries[i].value END;
			RETURN Missing
		END Get;

		PROCEDURE Has* (CONST key: ARRAY OF CHAR): BOOLEAN;
		BEGIN RETURN entries[Slot(key)].state = used
		END Has;

		PROCEDURE Put* (CONST key: ARRAY OF CHAR; value: SIZE);
		VAR i: SIZE;
		BEGIN
			i := Slot(key);
			IF entries[i].state # used THEN
				IF entries[i].state = empty THEN INC(occupied) END;
				entries[i].key := Strings.NewString(key);
				entries[i].state := used;
				INC(count)
			END;
			entries[i].value := value;
			IF occupied * 4 > LEN(entries) * 3 THEN Grow END
		END Put;

		(** Add increments the value, treating an absent key as zero, and returns the new one. *)
		PROCEDURE Add* (CONST key: ARRAY OF CHAR; delta: SIZE): SIZE;
		VAR v: SIZE;
		BEGIN
			v := Get(key);
			IF v = Missing THEN v := 0 END;
			INC(v, delta);
			Put(key, v);
			RETURN v
		END Add;

		PROCEDURE Delete* (CONST key: ARRAY OF CHAR);
		VAR i: SIZE;
		BEGIN
			i := Slot(key);
			IF entries[i].state = used THEN
				entries[i].state := deleted;
				entries[i].key := NIL;
				DEC(count)
			END
		END Delete;

		(** Iterate walks every live entry: start with slot = 0 and pass it back unchanged. *)
		PROCEDURE Iterate* (VAR slot: SIZE; VAR key: Strings.String; VAR value: SIZE): BOOLEAN;
		BEGIN
			WHILE slot < LEN(entries) DO
				IF entries[slot].state = used THEN
					key := entries[slot].key;
					value := entries[slot].value;
					INC(slot);
					RETURN TRUE
				END;
				INC(slot)
			END;
			RETURN FALSE
		END Iterate;

		PROCEDURE Grow;
		VAR old: EntryArray; i, target: SIZE;
		BEGIN
			old := entries;
			target := LEN(old);
			(* tombstones alone do not justify a bigger table, only live keys do *)
			IF count * 2 > target THEN target := target * 2 END;
			NEW(entries, target);
			count := 0; occupied := 0;
			FOR i := 0 TO LEN(old) - 1 DO
				IF old[i].state = used THEN Put(old[i].key^, old[i].value) END
			END
		END Grow;

	END Map;

	(** List is a growable array of strings: id -> token, and the merge tables. *)
	List* = OBJECT
	VAR
		items: Strings.StringArray;
		count-: SIZE;

		PROCEDURE &Init* (initial: SIZE);
		BEGIN
			IF initial < 4 THEN initial := 4 END;
			NEW(items, initial); count := 0
		END Init;

		PROCEDURE Add* (CONST s: ARRAY OF CHAR);
		VAR bigger: Strings.StringArray; i: SIZE;
		BEGIN
			IF count = LEN(items) THEN
				NEW(bigger, count * 2);
				FOR i := 0 TO count - 1 DO bigger[i] := items[i] END;
				items := bigger
			END;
			items[count] := Strings.NewString(s);
			INC(count)
		END Add;

		PROCEDURE Get* (i: SIZE): Strings.String;
		BEGIN
			ASSERT((i >= 0) & (i < count));
			RETURN items[i]
		END Get;

		PROCEDURE Set* (i: SIZE; CONST s: ARRAY OF CHAR);
		BEGIN
			ASSERT((i >= 0) & (i < count));
			items[i] := Strings.NewString(s)
		END Set;

	END List;

PROCEDURE RoundUpPowerOfTwo (n: SIZE): SIZE;
VAR p: SIZE;
BEGIN
	p := 8;
	WHILE p < n DO p := p * 2 END;
	RETURN p
END RoundUpPowerOfTwo;

(* FNV-1a over the bytes: cheap, and good enough for keys that are short byte strings. *)
PROCEDURE Hash (CONST key: ARRAY OF CHAR): SIZE;
VAR h: UNSIGNED32; i: SIZE;
BEGIN
	h := 2166136261;
	i := 0;
	WHILE key[i] # 0X DO
		h := UNSIGNED32(SET32(h) / SET32(UNSIGNED32(ORD(key[i]))));
		h := h * 16777619;
		INC(i)
	END;
	RETURN SIZE(h MOD 7FFFFFFFH)
END Hash;

PROCEDURE {TEST} TestMap*;
VAR m: Map; slot, value, seen: SIZE; key: Strings.String;
BEGIN
	NEW(m, 8);
	ASSERT(m.Get("absent") = Missing);
	m.Put("a", 1); m.Put("bb", 2); m.Put("ccc", 3);
	ASSERT(m.count = 3);
	ASSERT(m.Get("a") = 1);
	ASSERT(m.Get("ccc") = 3);
	m.Put("a", 9);
	ASSERT(m.count = 3);
	ASSERT(m.Get("a") = 9);
	ASSERT(m.Add("bb", 5) = 7);
	ASSERT(m.Add("new", 4) = 4);

	m.Delete("bb");
	ASSERT(m.count = 3);
	ASSERT(~m.Has("bb"));
	ASSERT(m.Get("ccc") = 3);		(* the probe must walk past the tombstone *)

	slot := 0; seen := 0;
	WHILE m.Iterate(slot, key, value) DO INC(seen) END;
	ASSERT(seen = 3)
END TestMap;

PROCEDURE {TEST} TestMapGrows*;
VAR m: Map; i: SIZE; key: ARRAY 32 OF CHAR;
BEGIN
	NEW(m, 8);
	FOR i := 0 TO 999 DO
		key := "k"; Strings.AppendInt(key, SIGNED32(i));
		m.Put(key, i * 3)
	END;
	ASSERT(m.count = 1000);
	FOR i := 0 TO 999 DO
		key := "k"; Strings.AppendInt(key, SIGNED32(i));
		ASSERT(m.Get(key) = i * 3)
	END
END TestMapGrows;

PROCEDURE {TEST} TestList*;
VAR l: List; i: SIZE;
BEGIN
	NEW(l, 4);
	FOR i := 0 TO 99 DO l.Add("x") END;
	ASSERT(l.count = 100);
	l.Set(50, "changed");
	ASSERT(l.Get(50)^ = "changed");
	ASSERT(l.Get(0)^ = "x")
END TestList;

END TritMap.
