import os
from build123d import *

def generate_cube(out_dir):
    # 1. cube.step: Box 10x10x10
    with BuildPart() as p:
        Box(10, 10, 10)
    p.part.export_step(os.path.join(out_dir, "cube.step"))
    print("Successfully generated: cube.step")

def generate_cylinder_with_hole(out_dir):
    # 2. cylinder_with_hole.step: Tests topology (loops, inner faces)
    with BuildPart() as p:
        Cylinder(radius=10, height=20)
        with BuildSketch(p.faces().sort_by(Axis.Z)[-1]):
            Circle(radius=5)
        extrude(amount=-20, mode=Mode.SUBTRACT)
    p.part.export_step(os.path.join(out_dir, "cylinder_with_hole.step"))
    print("Successfully generated: cylinder_with_hole.step")

def generate_assembly(out_dir):
    # 3. assembly.step: Multiple entities / hierarchy
    # Make a compound of two boxes
    with BuildPart() as p1:
        Box(10, 10, 10)
    with BuildPart() as p2:
        with Locations((20, 0, 0)):
            Box(10, 10, 10)
            
    # Combine into an assembly
    a = Compound.make_compound([p1.part.wrapped, p2.part.wrapped])
    import OCP.STEPControl
    writer = OCP.STEPControl.STEPControl_Writer()
    writer.Transfer(a, OCP.STEPControl.STEPControl_AsIs)
    writer.Write(os.path.join(out_dir, "assembly.step"))
    print("Successfully generated: assembly.step")

def generate_broken(out_dir):
    # 4. broken.step: Syntactically broken or geometrically invalid to fail parsing
    filepath = os.path.join(out_dir, "broken.step")
    with open(filepath, "w") as f:
        f.write("""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Testing broken STEP file'), '2;1');
FILE_NAME('broken.step', '2026-03-24', ('Author'), ('Organization'), 'preprocessor', 'originator', 'authorization');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2 { 1 2 10303 214 -1 1 5 4 }'));
ENDSEC;
DATA;
#1=INVALID_ENTITY('',$,(),.F.);
#2=APPLICATION_CONTEXT('Broken Context');
ENDSEC;
END-ISO-10303-21;
""")
    print("Successfully generated: broken.step")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data", "test_parts")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating validation test STEP files using build123d...")
    generate_cube(out_dir)
    generate_cylinder_with_hole(out_dir)
    generate_assembly(out_dir)
    generate_broken(out_dir)
    print("Done!")
