# Lab認証を解除する計画

## Context

本番環境 (bandscore.vercel.app) で `LAB_ACCESS_KEY` 環境変数が設定されていないため、Lab ページ (`/lab`) にアクセスできていない。認証システムを無効化して、全員がアクセス可能にする。

## 推奨アプローチ

**ミドルウェア無効化のみ（ファイルは残す）**

認証関連ファイルは将来の再利用のために残し、middleware の保護ロジックのみを無効化します。

## 変更するファイル

| ファイル | 変更内容 |
|---------|---------|
| `frontend/middleware.ts` | `/lab` ルートの保護ロジックを削除 |

## 実装手順

### 1. middleware.ts を変更

**変更前:**
```typescript
// Only protect /lab routes
if (pathname.startsWith('/lab')) {
    // Exception for login page and its children
    if (pathname.startsWith('/lab/login')) {
        return NextResponse.next();
    }

    // Check for access cookie
    const hasAccess = request.cookies.has('lab_access');

    if (!hasAccess) {
        const loginUrl = new URL('/lab/login', request.url);
        return NextResponse.redirect(loginUrl);
    }
}
```

**変更後:**
```typescript
// Lab protection disabled - allowing public access
// if (pathname.startsWith('/lab')) { ... }
```

または、`/lab` ルートの保護ロジックを完全に削除して、matcher config から `/lab` を除外。

### 2. 本番環境への反映

変更をコミットしてプッシュし、Vercel にデプロイします。

## 検証方法

1. 本番サイト `https://bandscore.vercel.app/lab` に直接アクセス
2. ログインページにリダイレクトされず、Lab ページが表示されることを確認
3. `https://bandscore.vercel.app/lab/training` も同様にアクセス可能であることを確認

## 変更されないファイル（将来の再利用のために残す）

- `frontend/app/lab/login/page.tsx` - ログインページ
- `frontend/app/api/lab/login/route.ts` - ログインAPI
- `frontend/.env.local` の `LAB_ACCESS_KEY` - ローカル開発用

## 戻す場合の方法

将来認証を再度有効化する場合は、middleware.ts の変更をロールバックするだけで復元可能です。