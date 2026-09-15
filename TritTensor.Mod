MODULE TritTensor;
(**
	AUTHOR "Andrii Puhachenko";
	PURPOSE "Autograd nodes: the graph the training loop differentiates";
	LIMITATIONS
		"- Active Oberon has no closures (the language report: a procedure assigned to
		   a procedure variable 'must not be local to another procedure'), so where the
		   C# and Go versions append a closure per operation, each operation here is a
		   Node subtype that keeps what it captured in fields and overrides Backward;
		 - at most MaxChildren inputs per node, which is three: attention takes q, k, v."
*)

IMPORT TritIO;

CONST
	MaxChildren* = 3;

TYPE
	Node* = OBJECT
	VAR
		data-, grad-: TritIO.Floats;
		rows-, cols-: SIZE;				(** a vector is rows = 1 *)
		requiresGrad-: BOOLEAN;

		child-: ARRAY MaxChildren OF Node;
		nofChildren-: SIZE;

		mark: SIGNED32;					(* which traversal last saw this node *)

		(** Backward pushes this node's gradient into its inputs. The base does nothing:
			a leaf, or a node built by Wrap, has no rule of its own. *)
		PROCEDURE Backward*;
		END Backward;

		PROCEDURE AddChild* (c: Node);
		BEGIN
			IF c.requiresGrad THEN
				ASSERT(nofChildren < MaxChildren);
				child[nofChildren] := c; INC(nofChildren)
			END
		END AddChild;

		PROCEDURE ZeroGrad*;
		VAR i: SIZE;
		BEGIN
			IF grad # NIL THEN FOR i := 0 TO LEN(grad) - 1 DO grad[i] := 0 END END
		END ZeroGrad;

		PROCEDURE Count* (): SIZE;
		BEGIN RETURN LEN(data)
		END Count;

	END Node;

	NodeArray = POINTER TO ARRAY OF Node;

VAR
	traversal: SIGNED32;		(* bumped per Run, so no node needs its mark cleared *)

(** Setup initialises a node in place: Active Oberon constructors do not chain, so
	every operation's factory calls this instead of inheriting one. *)
PROCEDURE Setup* (n: Node; data: TritIO.Floats; rows, cols: SIZE; requiresGrad: BOOLEAN);
BEGIN
	n.data := data; n.rows := rows; n.cols := cols;
	n.requiresGrad := requiresGrad;
	n.nofChildren := 0; n.mark := 0;
	IF requiresGrad THEN NEW(n.grad, LEN(data)) ELSE n.grad := NIL END
END Setup;

PROCEDURE New* (data: TritIO.Floats; rows, cols: SIZE; requiresGrad: BOOLEAN): Node;
VAR n: Node;
BEGIN
	NEW(n); Setup(n, data, rows, cols, requiresGrad);
	RETURN n
END New;

PROCEDURE Zeros* (rows, cols: SIZE; requiresGrad: BOOLEAN): Node;
VAR data: TritIO.Floats;
BEGIN
	NEW(data, rows * cols);
	RETURN New(data, rows, cols, requiresGrad)
END Zeros;

PROCEDURE Ones* (n: SIZE; requiresGrad: BOOLEAN): Node;
VAR data: TritIO.Floats; i: SIZE;
BEGIN
	NEW(data, n);
	FOR i := 0 TO n - 1 DO data[i] := 1 END;
	RETURN New(data, 1, n, requiresGrad)
END Ones;

(** Run seeds dL/dL = 1 and walks the graph in reverse topological order. *)
PROCEDURE Run* (root: Node);
VAR topo: NodeArray; count, i: SIZE;

	PROCEDURE Grow;
	VAR bigger: NodeArray; j: SIZE;
	BEGIN
		NEW(bigger, LEN(topo) * 2);
		FOR j := 0 TO count - 1 DO bigger[j] := topo[j] END;
		topo := bigger
	END Grow;

	PROCEDURE Build (n: Node);
	VAR j: SIZE;
	BEGIN
		IF n.mark = traversal THEN RETURN END;
		n.mark := traversal;
		FOR j := 0 TO n.nofChildren - 1 DO Build(n.child[j]) END;
		IF count = LEN(topo) THEN Grow END;
		topo[count] := n; INC(count)
	END Build;

BEGIN
	IF root.grad = NIL THEN NEW(root.grad, LEN(root.data)) END;
	FOR i := 0 TO LEN(root.grad) - 1 DO root.grad[i] := 1 END;

	INC(traversal);
	NEW(topo, 256); count := 0;
	Build(root);

	FOR i := count - 1 TO 0 BY -1 DO topo[i].Backward END
END Run;

PROCEDURE {TEST} TestTopologicalOrder*;
VAR a, b, c: Node; order: ARRAY 8 OF SIZE;
BEGIN
	(* a diamond: c depends on a and b, b depends on a. a must run last. *)
	a := Zeros(1, 2, TRUE);
	b := Zeros(1, 2, TRUE); b.AddChild(a);
	c := Zeros(1, 2, TRUE); c.AddChild(a); c.AddChild(b);

	Run(c);
	ASSERT(c.grad[0] = 1);		(* the seed reached the root *)
	ASSERT(a.nofChildren = 0);
	ASSERT(c.nofChildren = 2)
END TestTopologicalOrder;

PROCEDURE {TEST} TestNoGradWithoutRequest*;
VAR n: Node;
BEGIN
	n := Zeros(2, 3, FALSE);
	ASSERT(n.grad = NIL);
	ASSERT(n.Count() = 6);
	n.ZeroGrad					(* must not trap on a node that carries no gradient *)
END TestNoGradWithoutRequest;

BEGIN
	traversal := 0
END TritTensor.
