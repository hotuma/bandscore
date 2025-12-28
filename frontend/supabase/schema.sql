-- Create the waitlist table
create table public.waitlist (
  id uuid default gen_random_uuid() primary key,
  email text not null unique,
  created_at timestamptz default now()
);

-- Enable RLS (Row Level Security)
alter table public.waitlist enable row level security;

-- Policy: Allow service role (backend) to insert/select
-- Apps using service_role key bypass RLS, so this is just for safety/completeness if needed.
-- But strictly speaking, if we ONLY use service_role key from the backend action, we technically bypass RLS.
-- However, if you want to be explicit or if you might use client key later (not recommended for this):
-- create policy "Service Role Full Access" on public.waitlist for all to service_role using (true) with check (true);
