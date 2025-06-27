import os
import gc
import ppdeep
import androguard.misc
import androguard.util
import androguard.session
import pandas as pd


def extract_api_calls(dx):
    """Extract unique API calls from the APK."""
    api_calls = set()
    for cls in dx.get_classes():
        for method in cls.get_methods():
            for _, call, _ in method.get_xref_to():
                if call.is_external():
                    api_call = f"{call.get_class_name()[1:-1]}.{call.name}"
                    api_calls.add(api_call)
    return sorted(list(api_calls))


def extract_opcodes(dx):
    """Extract opcode statistics from the APK."""
    # The model was trained on 768-dimensional opcode vectors, but Androguard
    # extracts 256. We pad with zeros to match the model's expected input shape.
    opcodes_counts = [0] * 768
    for cls in dx.get_classes():
        for method in cls.get_methods():
            if method.is_external():
                continue
            try:
                m = method.get_method()
                if m.get_code():
                    for instruction in m.get_code().get_bc().get_instructions():
                        opcode = instruction.get_op_value()
                        if 0 <= opcode < 256:
                            opcodes_counts[opcode] += 1
            except Exception:
                # Ignore errors in single methods, not critical for overall analysis
                pass
    return opcodes_counts


def analyze_apk(apk_path):
    """Analyzes a single APK file and returns its features as a dictionary."""
    # Create a new session for each analysis to ensure isolation
    s = androguard.misc.get_default_session()

    try:
        # Set logger to critical to avoid excessive output from androguard
        # If this function is called multiple times, the log level is already set,
        # but if we do it again it will not raise an error.
        try:
            androguard.util.set_log("CRITICAL")
        except Exception:
            ...

        a, d, dx = androguard.misc.AnalyzeAPK(apk_path, session=s)

        apk_size = os.path.getsize(apk_path)
        apk_fuzzy_hash = ppdeep.hash_from_file(apk_path)
        fuzzy_hash_1 = (
            apk_fuzzy_hash.split(":")[1] if ":" in apk_fuzzy_hash else apk_fuzzy_hash
        )

        # This structure must match the features the model was trained on
        features = {
            "file_size": apk_size,
            "fuzzy_hash": fuzzy_hash_1,
            "activities_list": a.get_activities(),
            "services_list": a.get_services(),
            "receivers_list": a.get_receivers(),
            "permissions_list": a.get_permissions(),
            "api_calls_list": extract_api_calls(dx),
            "opcode_counts": extract_opcodes(dx),
        }

        # Clean up Androguard objects before finishing the session
        del a, d, dx

        return features, os.path.basename(apk_path)
    except Exception as e:
        error_msg = f"Error analyzing {os.path.basename(apk_path)}: {str(e)}"
        return None, error_msg
    finally:
        s.reset()
        del s

        gc.collect()


def features_dict_to_dataframe(features_dict):
    """Converts the features dictionary to a single-row DataFrame for the model."""
    df = pd.DataFrame([features_dict])

    # Ensure list-like columns are treated as objects so pandas doesn't
    # try to create multiple columns out of them.
    list_columns = [
        "activities_list",
        "services_list",
        "receivers_list",
        "permissions_list",
        "api_calls_list",
        "opcode_counts",
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df
