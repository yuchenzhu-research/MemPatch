from __future__ import annotations
import os
import sys
import glob
import importlib.util
import traceback
import inspect
import tempfile
import pathlib

def run_all() -> None:
    # Add src to python path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    test_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    passed_count = 0
    failed_count = 0
    
    for test_file in sorted(test_files):
        module_name = os.path.basename(test_file)[:-3]
        if module_name == "run_all":
            continue
            
        print(f"Running tests in {module_name}...")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        if spec is None or spec.loader is None:
            print(f"  Failed to load spec/loader for {module_name}")
            failed_count += 1
            continue
            
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  Failed to import/execute module {module_name}: {e}")
            traceback.print_exc()
            failed_count += 1
            continue
            
        # Find all test functions
        test_funcs = [getattr(module, name) for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]
        
        for func in test_funcs:
            func_name = func.__name__
            sig = inspect.signature(func)
            kwargs = {}
            temp_dirs = []
            
            if "tmp_path" in sig.parameters:
                tmpdir = tempfile.TemporaryDirectory()
                temp_dirs.append(tmpdir)
                kwargs["tmp_path"] = pathlib.Path(tmpdir.name)
                
            try:
                func(**kwargs)
                passed_count += 1
            except Exception as e:
                print(f"  {func_name}: FAILED: {e}")
                traceback.print_exc()
                failed_count += 1
            finally:
                for tmpdir in temp_dirs:
                    try:
                        tmpdir.cleanup()
                    except Exception:
                        pass

    print("-" * 40)
    print(f"Total passed: {passed_count}")
    print(f"Total failed: {failed_count}")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all()
