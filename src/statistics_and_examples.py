


"""
# PREDICTION EXAMPLE
test_comment = "You are such a loser! You'll regret everything you've done to me!"
encoding = tokenizer.encode_plus(
    test_comment,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt',
)
_, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
test_prediction = test_prediction.flatten().numpy()
for label, prediction in zip(settings.LABEL_COLUMNS, test_prediction):
    if prediction > settings.THRESHOLD:
        print(f"{label}: {prediction}")
"""

"""ESTADISTICAS"""
# Frecuencia por palabra (grafico)
"""
data[columns_names].sum().sort_values().plot(kind="barh")
plt.show()
"""

# Cantidad de tokens por palabra
"""
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
token_count_arr = []
for _, row in train_df.iterrows():
  token_count = len(tokenizer.encode(
    row["comment"],
    max_length=MAX_TOKEN_COUNT,
    truncation=True
  ))
  token_count_arr.append(token_count)
sns.histplot(token_count_arr)
plt.xlim([0, 100]);
plt.show()
"""

# Convierto valores de columnas de emociones en un array.
# data['emotions'] = data[data.columns[2:]].values.tolist()
# data = data[['comment', 'emotions']]
# print(data.head())