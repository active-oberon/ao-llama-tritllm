MODULE TritIO;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Array types and the .NET binary encoding both model files use";
	LIMITATIONS
		"- strings are written the way .NET's BinaryWriter writes them -- a 7-bit
		   encoded byte count followed by UTF-8 -- so that a model file produced by
		   the C# original loads here unchanged, and one written here loads there;
		 - FP16 is converted by hand because Active Oberon has no half type, and
		   every scale in a packed layer is stored in one."
*)

IMPORT Streams, Reals, Strings;

TYPE
	Floats* = POINTER TO ARRAY OF FLOAT32;
	Ints* = POINTER TO ARRAY OF SIGNED32;
	Words* = POINTER TO ARRAY OF UNSIGNED64;
	Bytes* = POINTER TO ARRAY OF CHAR;

(** WriteString writes .NET's length-prefixed form: a 7-bit encoded byte count, then UTF-8. *)
PROCEDURE WriteString* (w: Streams.Writer; CONST s: ARRAY OF CHAR);
VAR n, i: SIZE;
BEGIN
	n := Strings.Length(s);
	WHILE n >= 80H DO
		w.Char(CHR(n MOD 80H + 80H));
		n := n DIV 80H
	END;
	w.Char(CHR(n));
	FOR i := 0 TO Strings.Length(s) - 1 DO w.Char(s[i]) END
END WriteString;

PROCEDURE ReadString* (r: Streams.Reader; VAR s: ARRAY OF CHAR): BOOLEAN;
VAR n, shift, i: SIZE; b: CHAR;
BEGIN
	n := 0; shift := 1;
	REPEAT
		b := r.Get();
		n := n + (ORD(b) MOD 80H) * shift;
		shift := shift * 80H
	UNTIL (ORD(b) < 80H) OR (r.res # Streams.Ok);
	IF n >= LEN(s) THEN RETURN FALSE END;
	FOR i := 0 TO n - 1 DO s[i] := r.Get() END;
	s[n] := 0X;
	RETURN r.res = Streams.Ok
END ReadString;

(** WriteVarInt is the unsigned LEB128 the compact tokenizer stores merge ids in. *)
PROCEDURE WriteVarInt* (w: Streams.Writer; value: SIZE);
BEGIN
	ASSERT(value >= 0);
	WHILE value >= 80H DO
		w.Char(CHR(value MOD 80H + 80H));
		value := value DIV 80H
	END;
	w.Char(CHR(value))
END WriteVarInt;

PROCEDURE ReadVarInt* (r: Streams.Reader): SIZE;
VAR result, shift: SIZE; b: CHAR;
BEGIN
	result := 0; shift := 1;
	REPEAT
		b := r.Get();
		result := result + (ORD(b) MOD 80H) * shift;
		shift := shift * 80H
	UNTIL (ORD(b) < 80H) OR (r.res # Streams.Ok);
	RETURN result
END ReadVarInt;

(** WriteFloats writes an int32 count followed by that many raw FP32 values. *)
PROCEDURE WriteFloats* (w: Streams.Writer; data: Floats);
VAR i: SIZE;
BEGIN
	IF data = NIL THEN w.RawInt32(0); RETURN END;
	w.RawInt32(SIGNED32(LEN(data)));
	FOR i := 0 TO LEN(data) - 1 DO w.RawFloat32(data[i]) END
END WriteFloats;

PROCEDURE ReadFloats* (r: Streams.Reader): Floats;
VAR n: SIGNED32; i: SIZE; data: Floats;
BEGIN
	r.RawInt32(n);
	ASSERT(n >= 0);
	NEW(data, n);
	FOR i := 0 TO n - 1 DO r.RawFloat32(data[i]) END;
	RETURN data
END ReadFloats;

(** F32ToF16 rounds to nearest even, the way a C# (Half) cast does. *)
PROCEDURE F32ToF16* (f: FLOAT32): UNSIGNED16;
VAR bits, biased, mant, exp, half, rem, halfway, shift, i: SIGNED32; sign: UNSIGNED16;
BEGIN
	bits := Reals.Int(f);
	IF bits < 0 THEN sign := 8000H ELSE sign := 0 END;
	biased := (bits DIV 800000H) MOD 100H;
	mant := bits MOD 800000H;

	IF biased = 0FFH THEN
		IF mant # 0 THEN RETURN UNSIGNED16(sign + 7E00H) END;	(* NaN *)
		RETURN UNSIGNED16(sign + 7C00H)								(* infinity *)
	END;

	exp := biased - 127 + 15;

	IF exp >= 1FH THEN
		RETURN UNSIGNED16(sign + 7C00H)								(* out of half range *)
	ELSIF exp <= 0 THEN
		IF exp < -10 THEN RETURN sign END;							(* underflows to zero *)
		INC(mant, 800000H);
		shift := 14 - exp;
		halfway := 1; FOR i := 1 TO shift - 1 DO halfway := halfway * 2 END;
		half := mant DIV (halfway * 2);
		rem := mant MOD (halfway * 2);
		IF (rem > halfway) OR ((rem = halfway) & (half MOD 2 = 1)) THEN INC(half) END;
		RETURN UNSIGNED16(sign + half)
	ELSE
		half := exp * 400H + mant DIV 2000H;
		rem := mant MOD 2000H;
		IF (rem > 1000H) OR ((rem = 1000H) & (half MOD 2 = 1)) THEN INC(half) END;
		RETURN UNSIGNED16(sign + half)
	END
END F32ToF16;

PROCEDURE F16ToF32* (h: UNSIGNED16): FLOAT32;
VAR sign, exp, mant, bits: SIGNED32; v: FLOAT32;
BEGIN
	sign := (SIGNED32(h) DIV 8000H) MOD 2;
	exp := (SIGNED32(h) DIV 400H) MOD 20H;
	mant := SIGNED32(h) MOD 400H;

	IF exp = 0 THEN
		v := mant / 16777216.0;					(* mant * 2^-24, exact *)
		IF sign # 0 THEN v := -v END;
		RETURN v
	ELSIF exp = 1FH THEN
		bits := 7F800000H + mant * 2000H
	ELSE
		bits := (exp + 112) * 800000H + mant * 2000H
	END;
	IF sign # 0 THEN bits := bits + MIN(SIGNED32) END;	(* set the sign bit *)
	RETURN Reals.Real(bits)
END F16ToF32;

PROCEDURE {TEST} TestFloat16* ;
BEGIN
	ASSERT(F32ToF16(0.0) = 0000H);
	ASSERT(F32ToF16(1.0) = 3C00H);
	ASSERT(F32ToF16(-2.0) = 0C000H);
	ASSERT(F32ToF16(0.5) = 3800H);
	ASSERT(F32ToF16(65504.0) = 7BFFH);			(* largest finite half *)
	ASSERT(ABS(F16ToF32(3C00H) - 1.0) < 0.0001);
	ASSERT(ABS(F16ToF32(0C000H) + 2.0) < 0.0001);
	ASSERT(ABS(F16ToF32(F32ToF16(0.1)) - 0.1) < 0.001);
	ASSERT(ABS(F16ToF32(F32ToF16(-1234.0)) + 1234.0) < 1.0);
	ASSERT(F16ToF32(0400H) > 0.0);				(* smallest normal *)
	ASSERT(F16ToF32(0001H) > 0.0)				(* smallest subnormal *)
END TestFloat16;

END TritIO.
