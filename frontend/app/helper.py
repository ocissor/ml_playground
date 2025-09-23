import streamlit as st
from config import model_hyperparams_values

def generate_hyperparameters_widget(params, model_choice):
    for i, (param, value) in enumerate(params.items()):
        # Figure out param type from first non-None option
        options = model_hyperparams_values[model_choice][param]["options"]
        default = model_hyperparams_values[model_choice][param]["default"]

        param_type = None
        for val in options:
            if val is not None:
                param_type = type(val).__name__
                break

        # Base widget key (unique across model + param + type + loop index)
        base_key = f"{model_choice}_{param}_{param_type}_{i}"

        if param_type == "int":
            try:
                params[param] = st.slider(
                    f"{param} ({param_type})",
                    min_value=options[0] if options[0] is not None else options[1],
                    max_value=int(options[-1]),
                    step=1,
                    value=default,
                    key=f"{base_key}_slider"
                )
            except Exception:
                params[param] = st.number_input(
                    f"{param} ({param_type})",
                    min_value=0,
                    max_value=1,
                    step=1,
                    key=f"{base_key}_number"
                )

        elif param_type == "float":
            try:
                params[param] = st.slider(
                    f"{param} ({param_type})",
                    min_value=options[0] if options[0] is not None else options[1],
                    max_value=options[-1],
                    step=0.01,
                    value=default,
                    key=f"{base_key}_slider"
                )
            except Exception:
                params[param] = st.number_input(
                    f"{param} ({param_type})",
                    min_value=0,
                    max_value=1,
                    step=0.01,
                    key=f"{base_key}_number"
                )

        elif param_type == "str":
            params[param] = st.selectbox(
                f"{param} ({param_type})",
                options,
                key=f"{base_key}_select"
            )

        elif param_type == "bool":
            params[param] = st.selectbox(
                f"{param} ({param_type})",
                options,
                key=f"{base_key}_bool"
            )

        else:
            st.warning(f"⚠️ Unsupported parameter type: {param_type} for {param}")

    return params