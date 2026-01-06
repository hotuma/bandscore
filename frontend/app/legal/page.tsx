import React from 'react';

export default function LegalPage() {
    return (
        <div className="min-h-screen bg-neutral-950 text-white p-8 font-sans">
            <div className="max-w-3xl mx-auto space-y-12">
                <header className="border-b border-neutral-800 pb-6">
                    <h1 className="text-3xl font-bold">特定商取引法に基づく表記</h1>
                </header>

                <section className="space-y-8">
                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">事業者名</h2>
                        <p className="text-neutral-300">Bandscore</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">運営責任者</h2>
                        <p className="text-neutral-300">新海朋哉</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">所在地</h2>
                        <p className="text-neutral-300">愛知県岡崎市</p>
                        <p className="text-sm text-neutral-500">※請求があった場合には、遅滞なく開示いたします。</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">メールアドレス</h2>
                        <p className="text-neutral-300">swype1222@gmail.com</p>
                        <p className="text-sm text-neutral-500">※お問い合わせは原則メールにてお願いいたします。</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">販売価格</h2>
                        <p className="text-neutral-300">2980円（税込）</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">商品代金以外の必要料金</h2>
                        <p className="text-neutral-300">インターネット接続にかかる通信費等は、利用者のご負担となります。</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">支払い方法</h2>
                        <p className="text-neutral-300">クレジットカード決済（Stripe）</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">支払い時期</h2>
                        <p className="text-neutral-300">クレジットカード決済：ご購入時に即時決済されます。</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">商品の提供時期</h2>
                        <p className="text-neutral-300">決済完了後、即時利用可能となります。</p>
                        <p className="text-neutral-400 text-sm">（Early Access 機能・デジタルコンテンツとして提供）</p>
                    </div>

                    <div className="space-y-4">
                        <h2 className="text-lg font-semibold text-teal-400">返品・キャンセルについて</h2>
                        <div className="space-y-4 text-neutral-300">
                            <p>
                                デジタルコンテンツおよびオンラインサービスの性質上、<br />
                                決済完了後の返金・キャンセルには原則として対応しておりません。
                            </p>
                            <p>ただし、以下の場合には個別に対応いたします。</p>
                            <ul className="list-disc list-inside space-y-1 text-neutral-400 ml-4">
                                <li>システム障害等により、サービスが長期間利用できなかった場合</li>
                                <li>当方の重大な過失による不具合が確認された場合</li>
                            </ul>
                            <p>上記の場合は、メールにてご連絡ください。</p>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">動作環境</h2>
                        <p className="text-neutral-300">最新の Google Chrome / Safari / Edge 等のモダンブラウザを推奨します。</p>
                        <p className="text-neutral-400 text-sm">すべての環境での動作を保証するものではありません。</p>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-lg font-semibold text-teal-400">表現およびサービスに関する注意書き</h2>
                        <p className="text-neutral-300">
                            本サービスで提供する解析結果・生成内容は、<br />
                            正確性・完全性・特定目的への適合性を保証するものではありません。<br />
                            利用者の判断と責任においてご利用ください。
                        </p>
                    </div>
                </section>
            </div>
        </div>
    );
}
