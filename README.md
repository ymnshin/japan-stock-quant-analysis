# 日本株クオンツ分析：PER・ROEスクリーニングの再検証

[Qiita「成功する投資：トレーディングのサイエンス」](https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9)で紹介されている「PER 10倍以下・ROE 10%以上」の検証例を参考に、現在の `yfinance` API で再実装・再検証したものです。TOPIX500 の銘柄を月末にスクリーニングし、翌1か月の銘柄リターンから日経平均のリターンを引いた超過リターンのヒストグラムを作ります。

## 結論

今回のデータと検証方法では、PER・ROEのスクリーン条件に再現可能なプラスの優位性は確認できませんでした。

- 基準条件（PER 10倍以下・ROE 10%以上）の全期間の平均月次超過リターンは約 -0.30%、プラスになった観測の割合は約48.8%でした。
- 開発期間だけで選んだ最良条件は PER 11倍以下・ROE 5%以上でしたが、条件選択に使わなかった2025年3月～2026年5月では平均月次超過リターンが約 -2.24%でした。
- 基準条件も同じ未使用期間では約 -2.65%でした。
- どちらの未使用期間の平均値もブートストラップ95%信頼区間が0をまたいでおり、安定した正のエッジを示す結果ではありませんでした。

したがって、「割安かつ高ROE」という条件だけで将来の超過収益を得られる、とは本検証からは判断できません。なお、これはPERやROEがあらゆる市場・期間で無意味だと証明するものではなく、下記のデータ制約を含む今回の設計では有効性を確認できなかった、という結論です。

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

## 参照元

- blog_UKI, [「成功する投資：トレーディングのサイエンス」](https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9), Qiita, 2020年11月12日更新


## Market Journal

個人勉強用の市場観察ログを `market-journal/` に記録しています。日本株、米国株、為替、金利、原油、半導体・AIなどについて、Fact / Interpretation / Scenario / Falsification を分けて日次で残します。

- [Market Journal](market-journal/README.md)
