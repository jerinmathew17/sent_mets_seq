import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# ========== MODEL DEFINITIONS ========== #

class NewsImpactModel(nn.Module):
    def __init__(self, ticker_vocab_size, ticker_embed_dim=32):
        super(NewsImpactModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.ticker_embedding = nn.Embedding(ticker_vocab_size, ticker_embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(768 + ticker_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, ticker_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        ticker_embed = self.ticker_embedding(ticker_ids)
        combined = torch.cat([cls_token, ticker_embed], dim=1)
        return self.regressor(combined)


class StockLSTM(nn.Module):
    def __init__(self, input_size=801, hidden_size=64, output_size=1, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)

# ========== LOAD MODELS ========== #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load small zero-shot BART model
bart_model_name = "valhalla/distilbart-mnli-12-1"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = AutoModelForSequenceClassification.from_pretrained(bart_model_name).to(device)
bart_model.eval()

# Load DistilBERT-based impact model
ticker_vocab_size = 54
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

best_bert_model = NewsImpactModel(ticker_vocab_size)
best_bert_model.load_state_dict(torch.load("best_bert_model.pth", map_location=device))
best_bert_model.eval().to(device)

lstm_model = StockLSTM()
lstm_model.load_state_dict(torch.load("path_to_save_lstm_model.pth", map_location=device))
lstm_model.eval().to(device)

# ========== SECTOR TO TICKERS ========== #

sector_to_tickers = {
    "Commodities": ["OAL.BO", "AMBUJACEM.BO", "SASHWAT.BO"],
    "Consumer Discretionary": ["INRADIA.BO", "ADL.BO", "KAMATHOTEL.BO", "FORCEMOT.BO", "AMBER.BO", "CARTRADE.BO", "ORBTEXP.BO", "ZEEMEDIA.BO", "APOLLOTYRE.BO", "VINNY.BO", "RUPA.BO", "CEENIK.BO", "RETINA.BO", "GTPL.BO", "BLUECOAST.BO", "RAYMOND.BO", "NIVAKA.BO", "SHOPERSTOP.BO"],
    "Fast Moving Consumer Goods": ["BCLIND.BO", "AGRITECH.BO", "RANASUG.BO"],
    "Financial Services": ["SSPNFIN.BO", "KHANDSE.BO", "MANAPPURAM.BO", "RAJPUTANA.BO", "NIKKIGL.BO", "KIDUJA.BO", "OPTIFIN.BO", "KEYFINSERV.BO", "NSIL.BO"],
    "Healthcare": ["GLAND.BO", "IPCALAB.BO"],
    "Industrials": ["SPMLINFRA.BO", "JWL.BO", "PARACABLES.BO", "ORIENTCER.BO", "PREMIER.BO", "ORIENTLTD.BO", "PARAS.BO", "SWSOLAR.BO", "MADHUCON.BO", "WELSPLSOL.BO"],
    "Information Technology": ["3IINFOLTD.BO", "IZMO.BO"],
    "Services": ["CONCOR.BO", "RISAINTL.BO", "PROCLB.BO"],
    "Telecommunication": ["STLTECH.BO", "GTLINFRA.BO"],
    "Utilities": ["ADANIPOWER.BO", "SURANAT&P.BO"]
}

# ========== FUNCTIONS ========== #

def get_sector_from_headline(headline):
    labels = list(sector_to_tickers.keys())
    premise = headline
    hypothesis_templates = [f"This news is about {sector}." for sector in labels]

    inputs = bart_tokenizer([premise]*len(labels), hypothesis_templates, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        logits = bart_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 2]  # Entailment class

    top_idx = torch.argmax(probs).item()
    return labels[top_idx]

def get_past_7_trading_days(ticker, ref_date):
    ref_date = datetime.strptime(ref_date, "%Y-%m-%d")
    end_date = ref_date + timedelta(days=1)
    start_date = ref_date - timedelta(days=14)
    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    df = df[df['Open'].notnull()]
    df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open']
    return df.tail(7)

def get_actual_price_change(ticker, date):
    df = yf.download(ticker, start=date, end=(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
    if len(df) == 0:
        return None
    return float((df["Close"].iloc[0] - df["Open"].iloc[0]) / df["Open"].iloc[0])

def generate_embedding(headline, ticker_id, price_change_pct):
    inputs = distilbert_tokenizer(headline, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        output = best_bert_model.bert(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :]  # (1, 768)

    ticker_tensor = torch.tensor([ticker_id], dtype=torch.long, device=device)
    ticker_embedding = best_bert_model.ticker_embedding(ticker_tensor)  # (1, 32)

    price_tensor = torch.tensor([[price_change_pct]], dtype=torch.float32, device=device)  # (1, 1)

    final_embedding = torch.cat([cls_embedding, ticker_embedding, price_tensor], dim=1)  # (1, 801)
    return final_embedding.squeeze(0)  # (801,)

def predict_next_day_change(input_tensor):
    with torch.no_grad():
        return lstm_model(input_tensor).squeeze().item()

# ========== MAIN INFERENCE ========== #

def run_real_time_inference(headline, ref_date="2023-09-08"):
    print(f"\n📰 Headline: {headline}")
    
    sector = get_sector_from_headline(headline)
    print(f"📊 Predicted Sector: {sector}")
    
    for ticker_id, ticker in enumerate(sector_to_tickers[sector]):
        print(f"\n🔍 Processing Ticker: {ticker}")
        try:
            df = get_past_7_trading_days(ticker, ref_date)
            if len(df) < 7:
                print("⚠️ Not enough trading days. Skipping...")
                continue

            embeddings = [
                generate_embedding(headline, ticker_id, df["price_change_pct"].iloc[i])
                for i in range(7)
            ]

            tensor_input = torch.stack(embeddings).unsqueeze(0).to(device)  # (1, 7, 801)
            predicted_change = predict_next_day_change(tensor_input)

            actual_change = get_actual_price_change(ticker, "2023-09-11")

            print(f"📈 Predicted % Change for 2025-04-11: {predicted_change:.4f}")
            print(f"📉 Actual % Change for 2025-04-11:    {actual_change:.4f}" if actual_change is not None else "❓ Actual % Change not available yet")

        except Exception as e:
            print(f"❌ Error for ticker {ticker}: {e}")

# ========== RUN ========== #

headline_input = input("Enter News Headline: ")
run_real_time_inference(headline_input)