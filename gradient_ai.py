from gradientai import Gradient
import os
from telegram_mes import send_telegram_message

os.environ['GRADIENT_ACCESS_TOKEN'] = "BVMSieGjgncZ9wuWRzYAhgWYYQom6GGv"
os.environ['GRADIENT_WORKSPACE_ID'] = "8a03a633-3367-45ce-9ff0-266936a8c56a_workspace"

gradient = Gradient()
base = gradient.get_base_model(base_model_slug="nous-hermes2")
my_adapter = gradient.get_model_adapter(model_adapter_id="b89c2ae0-5d2b-455b-a561-7095ddf4d746_model_adapter")

def weapon_detected_g(weapon):
    query = f"There is {weapon}. What should I do?"
    templated_query = f"<s\n###Input:\n{query}\n\n### Response:\n"
    response = my_adapter.complete(query=templated_query, max_generated_token_count=500)
    out_text = response.generated_output
    print(f">There is a dangerous situation. {response.generated_output}\n\n")
    send_telegram_message(out_text) 