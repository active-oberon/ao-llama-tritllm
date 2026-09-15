MODULE TritAdam;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "AdamW: Adam with decoupled weight decay and global gradient clipping";
	LIMITATIONS
		"parameters are held in registration order rather than in a hash table: the
		 clipping norm sums every gradient, and floating-point addition is not
		 associative, so an unordered walk would make two identical runs differ."
*)

IMPORT Math, TritIO, TritTensor;

TYPE
	NodeArray = POINTER TO ARRAY OF TritTensor.Node;
	MomentArray = POINTER TO ARRAY OF TritIO.Floats;

	Optimizer* = OBJECT
	VAR
		learningRate*, beta1*, beta2*, epsilon*, weightDecay*, gradientClip*: FLOAT32;

		step: SIZE;
		beta1t, beta2t: FLOAT32;

		params: NodeArray;
		m, v: MomentArray;
		count-: SIZE;

		PROCEDURE &Init* (lr: FLOAT32);
		BEGIN
			learningRate := lr;
			beta1 := 0.9; beta2 := 0.999; epsilon := 1.0E-8;
			weightDecay := 0.01; gradientClip := 1.0;
			step := 0; beta1t := 1; beta2t := 1;
			NEW(params, 16); NEW(m, 16); NEW(v, 16); count := 0
		END Init;

		PROCEDURE Register* (list: NodeArray; n: SIZE);
		VAR i: SIZE;
		BEGIN
			FOR i := 0 TO n - 1 DO RegisterOne(list[i]) END
		END Register;

		PROCEDURE RegisterOne* (p: TritTensor.Node);
		VAR i: SIZE; biggerNodes: NodeArray; biggerM, biggerV: MomentArray;
		BEGIN
			IF ~p.requiresGrad THEN RETURN END;
			FOR i := 0 TO count - 1 DO
				IF params[i] = p THEN RETURN END
			END;
			IF count = LEN(params) THEN
				NEW(biggerNodes, count * 2); NEW(biggerM, count * 2); NEW(biggerV, count * 2);
				FOR i := 0 TO count - 1 DO
					biggerNodes[i] := params[i]; biggerM[i] := m[i]; biggerV[i] := v[i]
				END;
				params := biggerNodes; m := biggerM; v := biggerV
			END;
			params[count] := p;
			NEW(m[count], LEN(p.data)); NEW(v[count], LEN(p.data));
			INC(count)
		END RegisterOne;

		PROCEDURE Step*;
		VAR i, j: SIZE; bc1, bc2, totalNorm, scale, g, mHat, vHat, update: FLOAT32;
			p: TritTensor.Node; mi, vi: TritIO.Floats;
		BEGIN
			INC(step);
			beta1t := beta1t * beta1; beta2t := beta2t * beta2;
			bc1 := 1 / (1 - beta1t); bc2 := 1 / (1 - beta2t);

			IF gradientClip > 0 THEN
				totalNorm := 0;
				FOR i := 0 TO count - 1 DO
					p := params[i];
					FOR j := 0 TO LEN(p.grad) - 1 DO totalNorm := totalNorm + p.grad[j] * p.grad[j] END
				END;
				totalNorm := Math.sqrt(totalNorm);
				IF totalNorm > gradientClip THEN
					scale := gradientClip / (totalNorm + 1.0E-6);
					FOR i := 0 TO count - 1 DO
						p := params[i];
						FOR j := 0 TO LEN(p.grad) - 1 DO p.grad[j] := p.grad[j] * scale END
					END
				END
			END;

			FOR i := 0 TO count - 1 DO
				p := params[i]; mi := m[i]; vi := v[i];
				FOR j := 0 TO LEN(p.data) - 1 DO
					g := p.grad[j];
					mi[j] := beta1 * mi[j] + (1 - beta1) * g;
					vi[j] := beta2 * vi[j] + (1 - beta2) * g * g;
					mHat := mi[j] * bc1;
					vHat := vi[j] * bc2;
					update := learningRate * mHat / (Math.sqrt(vHat) + epsilon);
					IF weightDecay > 0 THEN
						update := update + learningRate * weightDecay * p.data[j]
					END;
					p.data[j] := p.data[j] - update
				END;
				p.ZeroGrad
			END
		END Step;

	END Optimizer;

PROCEDURE {TEST} TestDescendsAndClears*;
VAR o: Optimizer; p: TritTensor.Node; before: FLOAT32; i: SIZE;
BEGIN
	NEW(o, 0.1); o.weightDecay := 0; o.gradientClip := 0;
	p := TritTensor.Zeros(1, 4, TRUE);
	FOR i := 0 TO 3 DO p.data[i] := 1 END;
	o.RegisterOne(p);
	ASSERT(o.count = 1);
	o.RegisterOne(p);
	ASSERT(o.count = 1);					(* registering twice must not double the state *)

	before := p.data[0];
	FOR i := 0 TO 3 DO p.grad[i] := 1 END;
	o.Step;
	ASSERT(p.data[0] < before);				(* a positive gradient moves the weight down *)
	FOR i := 0 TO 3 DO ASSERT(p.grad[i] = 0) END		(* and the gradient is cleared *)
END TestDescendsAndClears;

PROCEDURE {TEST} TestClipsGlobalNorm*;
VAR o: Optimizer; p: TritTensor.Node; i: SIZE; moved: FLOAT32;
BEGIN
	(* with clipping on, a huge gradient must not move the weight further than a
	   modest one: that is the whole reason the clip is there *)
	NEW(o, 0.1); o.weightDecay := 0; o.gradientClip := 1.0;
	p := TritTensor.Zeros(1, 4, TRUE);
	o.RegisterOne(p);
	FOR i := 0 TO 3 DO p.grad[i] := 1000000.0 END;
	o.Step;
	moved := ABS(p.data[0]);
	ASSERT(moved < 0.2)						(* one Adam step is bounded by the learning rate *)
END TestClipsGlobalNorm;

END TritAdam.
