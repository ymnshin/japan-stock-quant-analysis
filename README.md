# PER 10倍以下・ROE 10%以上の日本株分析

Qiita 記事「成功する投資：トレーディングのサイエンス」の手順を、現在の `yfinance` API に合わせて再実装したものです。TOPIX500 の銘柄を月末にスクリーニングし、翌1か月の銘柄リターンから日経平均のリターンを引いた超過リターンのヒストグラムを作ります。

## 実行方法

```powershell
pip install -r requirements.txt
python src/analyze_value_quality.py
```

既定の分析期間は、現在の `yfinance` で取得できる年次財務データに合わせて 2022年4月から直近の完了月までです。記事と同じく、決算期末日を財務データの利用可能日として扱います。公表遅延を保守的に反映する場合は、たとえば次のように実行します。

```powershell
python src/analyze_value_quality.py --reporting-lag-days 90 --refresh
```

主な出力は `data/selected_value_quality_returns.csv` と `img/value_quality_return_histogram.png` です。`--refresh` を付けない場合、取得済みの価格・財務データを再利用します。

PER・ROE閾値を時系列外検証で探索する場合は、データ取得後に次を実行します。

```powershell
python src/search_value_quality_parameters.py
```

探索結果は `data/parameter_search_results.csv`、集計は `data/parameter_search_summary.json`、ヒートマップと分布比較は `img/parameter_search_*.png` に保存されます。最後の15か月は閾値選択に使わない検証期間として残します。

## 方法上の注意

- 銘柄ユニバースは 2021年4月30日時点の TOPIX500 です。この固定リストを過去へ適用するため、生存者バイアスがあります。
- PER は月末調整後終値 ÷ 直近年次 EPS、ROE は当期純利益 ÷ 自己資本です。
- 財務数値の訂正履歴や実際の公表日は `yfinance` から取得できないため、厳密な point-in-time バックテストではありません。
- 配当・分割の影響を反映するため、価格は `auto_adjust=True` の調整後終値を使います。
- 売買手数料、税金、スリッページは考慮していません。
