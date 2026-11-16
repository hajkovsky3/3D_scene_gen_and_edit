"""Manual ground-truth annotation of scene frames (docs/evaluation.md, Ručná anotácia).

A small local web tool: SAM 2 proposes a mask from clicked points, the annotator corrects it
with a brush and names the object instance (`chair#1`); the same name in several frames is
the same 3D object. Output in workspace/<scene>/annotations/, used by scenepipe.eval.
"""
