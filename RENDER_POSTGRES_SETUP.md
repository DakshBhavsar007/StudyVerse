# How to Set Up PostgreSQL on Render (The Solution)

Follow these exact steps to ensure your database data is safe and does not get deleted when your app restarts.

## Step 1: Create the Database on Render
1. Go to your [Render Dashboard](https://dashboard.render.com/).
2. Click the **"New +"** button at the top right.
3. Select **"PostgreSQL"**.
4. **Name**: Enter a name (e.g., `studyverse-db`).
5. **Database**: Leave this generic (e.g., `studyverse_db`) or use the default.
6. **User**: Leave as default.
7. **Region**: Choose the **same region** as your Web Service (e.g., Singapore, Oregon) to minimize latency.
8. **Plan**: Select **"Free"** (for hobby projects) or a paid plan for production.
9. Click **"Create Database"**.

## Step 2: Get the Connection URL
1. Wait a moment for the database to be created (status will turn green/Available).
2. On the database details page, look for **"Internal Database URL"**.
   - It looks like: `postgres://user:password@hostname:5432/dbname`
   - Use "Internal" if your Web Service is also on Render.
   - Use "External" if you are connecting from your local PC.
3. Click "Copy" to copy this URL.

## Step 3: Configure Your Web Service
1. Go back to your [Render Dashboard](https://dashboard.render.com/).
2. Click on your **Web Service** (the app you deployed).
3. Click on the **"Environment"** tab in the left sidebar.
4. Click **"Add Environment Variable"**.
5. **Key**: Enter `DATABASE_URL` (must be exactly this).
6. **Value**: Paste the **Internal Database URL** you copied in Step 2.
7. Click **"Save Changes"**.

## Step 4: Verify
1. Render will automatically redeploy your app when you save environment variables.
2. Watch the deployment logs. You should see logs indicating the database tables are being checked/migrated.
3. Once "Live", your app is now using the persistent PostgreSQL database!

> [!IMPORTANT]
> **Data Loss Warning**: Switching to this new database means you start fresh. Any users or data currently in your old SQLite file will remain in that file (which gets deleted anyway) and will NOT automatically move to the new Postgres database. You strictly start with a clean slate.
