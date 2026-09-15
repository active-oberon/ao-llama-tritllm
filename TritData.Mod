MODULE TritData;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Reads the corpus: a JSON array of strings, as the C# original stores it"
*)

IMPORT Files, JSON, Strings, TritMap;

(** LoadStrings returns an empty list for a missing or malformed file, the way the
	C# loader swallows a JsonException. *)
PROCEDURE LoadStrings* (CONST path: ARRAY OF CHAR): TritMap.List;
VAR list: TritMap.List; f: Files.File; r: Files.Reader; value: JSON.Value;
	array: JSON.Array; item: JSON.Item; text: Strings.String; length, i: SIZE;
BEGIN
	NEW(list, 16);
	f := Files.Old(path);
	IF f = NIL THEN RETURN list END;

	length := Files.Length(f);
	IF length <= 0 THEN RETURN list END;
	NEW(text, length + 1);
	Files.OpenReader(r, f, 0);
	FOR i := 0 TO length - 1 DO text[i] := r.Get() END;
	text[length] := 0X;

	IF ~JSON.Parse(text^, value) THEN RETURN list END;
	IF ~(value IS JSON.Array) THEN RETURN list END;

	array := value(JSON.Array);
	item := array.items;
	WHILE item # NIL DO
		IF item.value IS JSON.String THEN list.Add(item.value(JSON.String).value^) END;
		item := item.next
	END;
	RETURN list
END LoadStrings;

END TritData.
