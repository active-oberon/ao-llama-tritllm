MODULE TritGrad;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Differentiable operations: one Node subtype per rule";
	LIMITATIONS
		"the C# original also carries Gelu and a standalone Softmax; nothing on the
		 Llama path calls either, so they are not ported."
*)

IMPORT Math, TritIO, TritConfig, TritPacker, TritTensor;

TYPE
	Node = TritTensor.Node;

	AddNode = OBJECT (TritTensor.Node)
	VAR a, b: Node;
		PROCEDURE Backward*;
		VAR i: SIZE;
		BEGIN
			IF a.requiresGrad & (a.grad # NIL) THEN
				FOR i := 0 TO LEN(a.grad) - 1 DO a.grad[i] := a.grad[i] + grad[i] END
			END;
			IF b.requiresGrad & (b.grad # NIL) THEN
				FOR i := 0 TO LEN(b.grad) - 1 DO b.grad[i] := b.grad[i] + grad[i] END
			END
		END Backward;
	END AddNode;

	MulNode = OBJECT (TritTensor.Node)
	VAR a, b: Node;
		PROCEDURE Backward*;
		VAR i: SIZE;
		BEGIN
			IF a.requiresGrad & (a.grad # NIL) THEN
				FOR i := 0 TO LEN(a.grad) - 1 DO a.grad[i] := a.grad[i] + grad[i] * b.data[i] END
			END;
			IF b.requiresGrad & (b.grad # NIL) THEN
				FOR i := 0 TO LEN(b.grad) - 1 DO b.grad[i] := b.grad[i] + grad[i] * a.data[i] END
			END
		END Backward;
	END MulNode;

	MatMulNode = OBJECT (TritTensor.Node)
	VAR a, b: Node; M, K, N: SIZE;
		PROCEDURE Backward*;
		VAR i, j, k: SIZE; g: FLOAT32;
		BEGIN
			IF a.requiresGrad & (a.grad # NIL) THEN
				FOR i := 0 TO M - 1 DO
					FOR k := 0 TO K - 1 DO
						g := 0;
						FOR j := 0 TO N - 1 DO g := g + grad[i * N + j] * b.data[k * N + j] END;
						a.grad[i * K + k] := a.grad[i * K + k] + g
					END
				END
			END;
			IF b.requiresGrad & (b.grad # NIL) THEN
				FOR k := 0 TO K - 1 DO
					FOR j := 0 TO N - 1 DO
						g := 0;
						FOR i := 0 TO M - 1 DO g := g + grad[i * N + j] * a.data[i * K + k] END;
						b.grad[k * N + j] := b.grad[k * N + j] + g
					END
				END
			END
		END Backward;
	END MatMulNode;

	RmsNormNode = OBJECT (TritTensor.Node)
	VAR x, gamma: Node; seq, dim: SIZE; rmsInv: TritIO.Floats;
		PROCEDURE Backward*;
		VAR s, d: SIZE; gradSum, g, normed, scale: FLOAT32;
		BEGIN
			FOR s := 0 TO seq - 1 DO
				gradSum := 0;
				FOR d := 0 TO dim - 1 DO
					g := grad[s * dim + d];
					normed := x.data[s * dim + d] * rmsInv[s];
					IF gamma.requiresGrad & (gamma.grad # NIL) THEN
						gamma.grad[d] := gamma.grad[d] + g * normed
					END;
					gradSum := gradSum + g * gamma.data[d] * normed
				END;
				IF x.requiresGrad & (x.grad # NIL) THEN
					scale := rmsInv[s] / dim;
					FOR d := 0 TO dim - 1 DO
						g := grad[s * dim + d];
						normed := x.data[s * dim + d] * rmsInv[s];
						x.grad[s * dim + d] := x.grad[s * dim + d]
							+ scale * (dim * g * gamma.data[d] - normed * gradSum)
					END
				END
			END
		END Backward;
	END RmsNormNode;

	LogSoftmaxNode = OBJECT (TritTensor.Node)
	VAR x: Node; rows, columns: SIZE; probs: TritIO.Floats;
		PROCEDURE Backward*;
		VAR r, c, offset: SIZE; sumGrad: FLOAT32;
		BEGIN
			IF (x.grad = NIL) THEN RETURN END;
			FOR r := 0 TO rows - 1 DO
				offset := r * columns;
				sumGrad := 0;
				FOR c := 0 TO columns - 1 DO sumGrad := sumGrad + grad[offset + c] END;
				FOR c := 0 TO columns - 1 DO
					x.grad[offset + c] := x.grad[offset + c] + grad[offset + c] - probs[offset + c] * sumGrad
				END
			END
		END Backward;
	END LogSoftmaxNode;

	CrossEntropyNode = OBJECT (TritTensor.Node)
	VAR logProbs: Node; target: TritIO.Floats; rows: SIZE;
		PROCEDURE Backward*;
		VAR i: SIZE; scale: FLOAT32;
		BEGIN
			IF logProbs.grad = NIL THEN RETURN END;
			scale := 1 / FLOAT32(rows);
			FOR i := 0 TO LEN(logProbs.data) - 1 DO
				logProbs.grad[i] := logProbs.grad[i] - target[i] * scale
			END
		END Backward;
	END CrossEntropyNode;

	SiluNode = OBJECT (TritTensor.Node)
	VAR x: Node; sigmoid: TritIO.Floats;
		PROCEDURE Backward*;
		VAR i: SIZE; s: FLOAT32;
		BEGIN
			IF x.grad = NIL THEN RETURN END;
			FOR i := 0 TO LEN(x.data) - 1 DO
				s := sigmoid[i];
				x.grad[i] := x.grad[i] + grad[i] * (s + x.data[i] * s * (1 - s))
			END
		END Backward;
	END SiluNode;

	RoPENode = OBJECT (TritTensor.Node)
	VAR x: Node; seq, dim, heads, headDim, half: SIZE; cos, sin: TritIO.Floats;
		PROCEDURE Backward*;
		VAR pos, h, d, headStart, i1, i2, cs: SIZE; c, s, g1, g2: FLOAT32;
		BEGIN
			IF x.grad = NIL THEN RETURN END;
			FOR pos := 0 TO seq - 1 DO
				FOR h := 0 TO heads - 1 DO
					headStart := pos * dim + h * headDim;
					FOR d := 0 TO half - 1 DO
						cs := pos * half + d;
						c := cos[cs]; s := sin[cs];
						i1 := headStart + d; i2 := i1 + half;
						g1 := grad[i1]; g2 := grad[i2];
						x.grad[i1] := x.grad[i1] + g1 * c + g2 * s;
						x.grad[i2] := x.grad[i2] - g1 * s + g2 * c
					END
				END
			END
		END Backward;
	END RoPENode;

	TransposeNode = OBJECT (TritTensor.Node)
	VAR w: Node; wRows, wCols: SIZE;
		PROCEDURE Backward*;
		VAR i, j: SIZE;
		BEGIN
			IF w.grad = NIL THEN RETURN END;
			FOR i := 0 TO wRows - 1 DO
				FOR j := 0 TO wCols - 1 DO
					w.grad[i * wCols + j] := w.grad[i * wCols + j] + grad[j * wRows + i]
				END
			END
		END Backward;
	END TransposeNode;

	QuantizeNode = OBJECT (TritTensor.Node)
	VAR w: Node;
		PROCEDURE Backward*;
		VAR i: SIZE;
		BEGIN
			IF w.grad = NIL THEN RETURN END;
			FOR i := 0 TO LEN(w.grad) - 1 DO w.grad[i] := w.grad[i] + grad[i] END
		END Backward;
	END QuantizeNode;

PROCEDURE Add* (a, b: Node): Node;
VAR n: AddNode; out: TritIO.Floats; i: SIZE;
BEGIN
	ASSERT(LEN(a.data) = LEN(b.data));
	NEW(out, LEN(a.data));
	FOR i := 0 TO LEN(out) - 1 DO out[i] := a.data[i] + b.data[i] END;

	NEW(n);
	TritTensor.Setup(n, out, a.rows, a.cols, a.requiresGrad OR b.requiresGrad);
	n.a := a; n.b := b; n.AddChild(a); n.AddChild(b);
	RETURN n
END Add;

PROCEDURE Mul* (a, b: Node): Node;
VAR n: MulNode; out: TritIO.Floats; i: SIZE;
BEGIN
	ASSERT(LEN(a.data) = LEN(b.data));
	NEW(out, LEN(a.data));
	FOR i := 0 TO LEN(out) - 1 DO out[i] := a.data[i] * b.data[i] END;

	NEW(n);
	TritTensor.Setup(n, out, a.rows, a.cols, a.requiresGrad OR b.requiresGrad);
	n.a := a; n.b := b; n.AddChild(a); n.AddChild(b);
	RETURN n
END Mul;

(** MatMul is [M,K] x [K,N] = [M,N]. *)
PROCEDURE MatMul* (a, b: Node): Node;
VAR n: MatMulNode; out: TritIO.Floats; M, K, N, i, j, k: SIZE; sum: FLOAT32;
BEGIN
	M := a.rows; K := a.cols; N := b.cols;
	ASSERT(K = b.rows);
	NEW(out, M * N);
	FOR i := 0 TO M - 1 DO
		FOR j := 0 TO N - 1 DO
			sum := 0;
			FOR k := 0 TO K - 1 DO sum := sum + a.data[i * K + k] * b.data[k * N + j] END;
			out[i * N + j] := sum
		END
	END;

	NEW(n);
	TritTensor.Setup(n, out, M, N, a.requiresGrad OR b.requiresGrad);
	n.a := a; n.b := b; n.M := M; n.K := K; n.N := N;
	n.AddChild(a); n.AddChild(b);
	RETURN n
END MatMul;

(** RmsNorm normalizes over the last dimension: x is [seq, dim], gamma is [dim]. *)
PROCEDURE RmsNorm* (x, gamma: Node): Node;
VAR n: RmsNormNode; out, rmsInv: TritIO.Floats; seq, dim, s, d: SIZE; sumSq, v: FLOAT32;
BEGIN
	seq := x.rows; dim := x.cols;
	ASSERT(LEN(gamma.data) = dim);
	NEW(out, LEN(x.data)); NEW(rmsInv, seq);

	FOR s := 0 TO seq - 1 DO
		sumSq := 0;
		FOR d := 0 TO dim - 1 DO v := x.data[s * dim + d]; sumSq := sumSq + v * v END;
		rmsInv[s] := 1 / Math.sqrt(sumSq / dim + 1.0E-5);
		FOR d := 0 TO dim - 1 DO
			out[s * dim + d] := x.data[s * dim + d] * rmsInv[s] * gamma.data[d]
		END
	END;

	NEW(n);
	TritTensor.Setup(n, out, seq, dim, x.requiresGrad OR gamma.requiresGrad);
	n.x := x; n.gamma := gamma; n.seq := seq; n.dim := dim; n.rmsInv := rmsInv;
	n.AddChild(x); n.AddChild(gamma);
	RETURN n
END RmsNorm;

PROCEDURE LogSoftmax* (x: Node): Node;
VAR n: LogSoftmaxNode; out, probs: TritIO.Floats; rows, columns, r, c, offset: SIZE;
	maxValue, sum, logSum, e: FLOAT32;
BEGIN
	rows := x.rows; columns := x.cols;
	NEW(out, LEN(x.data)); NEW(probs, LEN(x.data));

	FOR r := 0 TO rows - 1 DO
		offset := r * columns;
		maxValue := x.data[offset];
		FOR c := 1 TO columns - 1 DO
			IF x.data[offset + c] > maxValue THEN maxValue := x.data[offset + c] END
		END;
		sum := 0;
		FOR c := 0 TO columns - 1 DO
			e := Math.exp(x.data[offset + c] - maxValue);
			probs[offset + c] := e; sum := sum + e
		END;
		logSum := Math.ln(sum + 1.0E-10);
		FOR c := 0 TO columns - 1 DO
			probs[offset + c] := probs[offset + c] / (sum + 1.0E-10);
			out[offset + c] := x.data[offset + c] - maxValue - logSum
		END
	END;

	NEW(n);
	TritTensor.Setup(n, out, rows, columns, x.requiresGrad);
	n.x := x; n.rows := rows; n.columns := columns; n.probs := probs;
	n.AddChild(x);
	RETURN n
END LogSoftmax;

(** CrossEntropy is -sum(target * log_softmax(x)) / rows, over one-hot or smoothed rows. *)
PROCEDURE CrossEntropy* (x: Node; target: TritIO.Floats): Node;
VAR n: CrossEntropyNode; logProbs: Node; out: TritIO.Floats; loss: FLOAT32; i: SIZE;
BEGIN
	ASSERT(LEN(x.data) = LEN(target));
	logProbs := LogSoftmax(x);

	loss := 0;
	FOR i := 0 TO LEN(logProbs.data) - 1 DO loss := loss - target[i] * logProbs.data[i] END;
	loss := loss / FLOAT32(x.rows);

	NEW(out, 1); out[0] := loss;
	NEW(n);
	TritTensor.Setup(n, out, 1, 1, x.requiresGrad);
	n.logProbs := logProbs; n.target := target; n.rows := x.rows;
	n.AddChild(logProbs);
	RETURN n
END CrossEntropy;

PROCEDURE Silu* (x: Node): Node;
VAR n: SiluNode; out, sigmoid: TritIO.Floats; i: SIZE; s: FLOAT32;
BEGIN
	NEW(out, LEN(x.data)); NEW(sigmoid, LEN(x.data));
	FOR i := 0 TO LEN(x.data) - 1 DO
		s := 1 / (1 + Math.exp(-x.data[i]));
		sigmoid[i] := s;
		out[i] := x.data[i] * s
	END;

	NEW(n);
	TritTensor.Setup(n, out, x.rows, x.cols, x.requiresGrad);
	n.x := x; n.sigmoid := sigmoid; n.AddChild(x);
	RETURN n
END Silu;

(** ApplyRoPE rotates each head's first half against its second. Training always
	starts at position zero, so there is no cache offset here. *)
PROCEDURE ApplyRoPE* (x: Node; heads, headDim: SIZE; theta: FLOAT32): Node;
VAR n: RoPENode; out, cos, sin: TritIO.Floats; seq, dim, half, pos, h, d, headStart, i1, i2, cs: SIZE;
	invFreq, angle, c, s, x1, x2: FLOAT32;
BEGIN
	seq := x.rows; dim := x.cols;
	ASSERT(headDim MOD 2 = 0);
	ASSERT(dim = heads * headDim);
	half := headDim DIV 2;

	NEW(out, LEN(x.data)); NEW(cos, seq * half); NEW(sin, seq * half);

	FOR pos := 0 TO seq - 1 DO
		FOR d := 0 TO half - 1 DO
			invFreq := 1 / Math.exp(2.0 * d / headDim * Math.ln(theta));
			angle := pos * invFreq;
			cs := pos * half + d;
			cos[cs] := Math.cos(angle); sin[cs] := Math.sin(angle)
		END
	END;

	FOR pos := 0 TO seq - 1 DO
		FOR h := 0 TO heads - 1 DO
			headStart := pos * dim + h * headDim;
			FOR d := 0 TO half - 1 DO
				cs := pos * half + d;
				c := cos[cs]; s := sin[cs];
				i1 := headStart + d; i2 := i1 + half;
				x1 := x.data[i1]; x2 := x.data[i2];
				out[i1] := x1 * c - x2 * s;
				out[i2] := x1 * s + x2 * c
			END
		END
	END;

	NEW(n);
	TritTensor.Setup(n, out, seq, dim, x.requiresGrad);
	n.x := x; n.seq := seq; n.dim := dim; n.heads := heads; n.headDim := headDim;
	n.half := half; n.cos := cos; n.sin := sin;
	n.AddChild(x);
	RETURN n
END ApplyRoPE;

PROCEDURE Transpose* (w: Node): Node;
VAR n: TransposeNode; out: TritIO.Floats; rows, cols, i, j: SIZE;
BEGIN
	rows := w.rows; cols := w.cols;
	NEW(out, rows * cols);
	FOR i := 0 TO rows - 1 DO
		FOR j := 0 TO cols - 1 DO out[j * rows + i] := w.data[i * cols + j] END
	END;

	NEW(n);
	TritTensor.Setup(n, out, cols, rows, w.requiresGrad);
	n.w := w; n.wRows := rows; n.wCols := cols; n.AddChild(w);
	RETURN n
END Transpose;

(** Quantize forwards the packed value and passes the gradient through unchanged --
	the straight-through estimator that makes quantization-aware training work. *)
PROCEDURE Quantize* (w: Node; planes: SIZE; cfg: TritConfig.Config): Node;
VAR n: QuantizeNode; packer: TritPacker.Packer;
BEGIN
	packer := TritPacker.FromFloat(w.data, w.rows, w.cols, planes, cfg.groupSize, cfg.scaleMode, 0);

	NEW(n);
	TritTensor.Setup(n, packer.ToFloat(), w.rows, w.cols, w.requiresGrad);
	n.w := w; n.AddChild(w);
	RETURN n
END Quantize;

(* ---- gradient checks: every rule below is compared against a central difference ---- *)

CONST
	eps = 0.01;			(* FP32 activations: a smaller step drowns in rounding *)
	tolerance = 0.02;

PROCEDURE Close (analytic, numeric: FLOAT32): BOOLEAN;
VAR scale: FLOAT32;
BEGIN
	scale := ABS(analytic);
	IF scale < 1 THEN scale := 1 END;
	RETURN ABS(numeric - analytic) / scale <= tolerance
END Close;

PROCEDURE Copy (src: TritIO.Floats): TritIO.Floats;
VAR out: TritIO.Floats; i: SIZE;
BEGIN
	NEW(out, LEN(src));
	FOR i := 0 TO LEN(src) - 1 DO out[i] := src[i] END;
	RETURN out
END Copy;

PROCEDURE Filled (n: SIZE; seed: SIZE): TritIO.Floats;
VAR v: TritIO.Floats; i: SIZE;
BEGIN
	NEW(v, n);
	FOR i := 0 TO n - 1 DO v[i] := ((i * 7 + seed * 13) MOD 11) / 5.0 - 1.0 END;
	RETURN v
END Filled;

PROCEDURE {TEST} TestRmsNormCrossEntropyGradient*;
VAR xData, gammaData, target: TritIO.Floats; x, gamma: Node;
	i, probe, s: SIZE; up, down, numeric: FLOAT32;
CONST seq = 3; dim = 6;

	PROCEDURE Loss (): FLOAT32;
	VAR a, b: Node;
	BEGIN
		a := TritTensor.New(Copy(xData), seq, dim, TRUE);
		b := TritTensor.New(Copy(gammaData), 1, dim, TRUE);
		RETURN CrossEntropy(RmsNorm(a, b), target).data[0]
	END Loss;

BEGIN
	xData := Filled(seq * dim, 1);
	gammaData := Filled(dim, 2);
	FOR i := 0 TO dim - 1 DO gammaData[i] := 1 + 0.1 * i END;
	NEW(target, seq * dim);
	FOR s := 0 TO seq - 1 DO target[s * dim + s MOD dim] := 1 END;

	x := TritTensor.New(xData, seq, dim, TRUE);
	gamma := TritTensor.New(gammaData, 1, dim, TRUE);
	TritTensor.Run(CrossEntropy(RmsNorm(x, gamma), target));

	FOR probe := 0 TO 2 DO
		i := probe * (seq * dim DIV 3);
		xData[i] := xData[i] + eps; up := Loss();
		xData[i] := xData[i] - 2 * eps; down := Loss();
		xData[i] := xData[i] + eps;
		numeric := (up - down) / (2 * eps);
		ASSERT(Close(x.grad[i], numeric))
	END;

	FOR probe := 0 TO 2 DO
		i := probe * (dim DIV 3);
		gammaData[i] := gammaData[i] + eps; up := Loss();
		gammaData[i] := gammaData[i] - 2 * eps; down := Loss();
		gammaData[i] := gammaData[i] + eps;
		numeric := (up - down) / (2 * eps);
		ASSERT(Close(gamma.grad[i], numeric))
	END
END TestRmsNormCrossEntropyGradient;

PROCEDURE {TEST} TestSiluAndMulGradient*;
VAR aData, bData, target: TritIO.Floats; a: Node; i, probe: SIZE; up, down, numeric: FLOAT32;
CONST rows = 2; cols = 4;

	PROCEDURE Loss (): FLOAT32;
	VAR p, q: Node;
	BEGIN
		p := TritTensor.New(Copy(aData), rows, cols, TRUE);
		q := TritTensor.New(Copy(bData), rows, cols, TRUE);
		RETURN CrossEntropy(Mul(Silu(p), q), target).data[0]
	END Loss;

BEGIN
	aData := Filled(rows * cols, 3);
	bData := Filled(rows * cols, 5);
	NEW(target, rows * cols);
	target[1] := 1; target[cols + 2] := 1;

	a := TritTensor.New(aData, rows, cols, TRUE);
	TritTensor.Run(CrossEntropy(Mul(Silu(a),
		TritTensor.New(Copy(bData), rows, cols, TRUE)), target));

	FOR probe := 0 TO 2 DO
		i := probe * 3;
		aData[i] := aData[i] + eps; up := Loss();
		aData[i] := aData[i] - 2 * eps; down := Loss();
		aData[i] := aData[i] + eps;
		numeric := (up - down) / (2 * eps);
		ASSERT(Close(a.grad[i], numeric))
	END
END TestSiluAndMulGradient;

PROCEDURE {TEST} TestMatMulAndRoPEGradient*;
VAR xData, wData, target: TritIO.Floats; x: Node; i, probe: SIZE; up, down, numeric: FLOAT32;
CONST seq = 3; heads = 2; headDim = 2; dim = heads * headDim;

	PROCEDURE Loss (): FLOAT32;
	VAR p, w: Node;
	BEGIN
		p := TritTensor.New(Copy(xData), seq, dim, TRUE);
		w := TritTensor.New(Copy(wData), dim, dim, TRUE);
		RETURN CrossEntropy(MatMul(ApplyRoPE(p, heads, headDim, 10000.0), w), target).data[0]
	END Loss;

BEGIN
	xData := Filled(seq * dim, 7);
	wData := Filled(dim * dim, 11);
	NEW(target, seq * dim);
	target[0] := 1; target[dim + 1] := 1; target[2 * dim + 2] := 1;

	x := TritTensor.New(xData, seq, dim, TRUE);
	TritTensor.Run(CrossEntropy(MatMul(ApplyRoPE(x, heads, headDim, 10000.0),
		TritTensor.New(Copy(wData), dim, dim, TRUE)), target));

	FOR probe := 0 TO 2 DO
		i := probe * 4;
		xData[i] := xData[i] + eps; up := Loss();
		xData[i] := xData[i] - 2 * eps; down := Loss();
		xData[i] := xData[i] + eps;
		numeric := (up - down) / (2 * eps);
		ASSERT(Close(x.grad[i], numeric))
	END
END TestMatMulAndRoPEGradient;

PROCEDURE {TEST} TestQuantizeIsStraightThrough*;
VAR wData: TritIO.Floats; w, q: Node; cfg: TritConfig.Config; i: SIZE; differs: BOOLEAN;
BEGIN
	NEW(cfg); cfg.groupSize := 4; cfg.scaleMode := TritConfig.ScalePerGroup;
	wData := Filled(8, 13);
	w := TritTensor.New(wData, 2, 4, TRUE);
	q := Quantize(w, 1, cfg);

	(* forward is the packed value: a grid of at most three levels per group *)
	differs := FALSE;
	FOR i := 0 TO 7 DO
		IF q.data[i] # wData[i] THEN differs := TRUE END
	END;
	ASSERT(differs);

	FOR i := 0 TO 7 DO q.grad[i] := i + 1 END;
	q.Backward;
	(* backward is the identity: that is the whole point of the estimator *)
	FOR i := 0 TO 7 DO ASSERT(w.grad[i] = i + 1) END
END TestQuantizeIsStraightThrough;

PROCEDURE {TEST} TestTransposeGradient*;
VAR wData: TritIO.Floats; w, t: Node; i, j: SIZE;
CONST rows = 2; cols = 3;
BEGIN
	wData := Filled(rows * cols, 17);
	w := TritTensor.New(wData, rows, cols, TRUE);
	t := Transpose(w);
	ASSERT(t.rows = cols); ASSERT(t.cols = rows);
	FOR i := 0 TO rows - 1 DO
		FOR j := 0 TO cols - 1 DO ASSERT(t.data[j * rows + i] = wData[i * cols + j]) END
	END;

	FOR i := 0 TO rows * cols - 1 DO t.grad[i] := i + 1 END;
	t.Backward;
	FOR i := 0 TO rows - 1 DO
		FOR j := 0 TO cols - 1 DO
			ASSERT(w.grad[i * cols + j] = j * rows + i + 1)
		END
	END
END TestTransposeGradient;



END TritGrad.
