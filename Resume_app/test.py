try:
    from ibm_watsonx_ai import Credentials, APIClient
    from ibm_watsonx_ai.foundation_models import Model, ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
    print("All imports successful")

except Exception as e:
    print("Import error:", e)