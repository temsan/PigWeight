-- Создание схемы для системы отслеживания свиней

-- Таблица актов взвешивания
CREATE TABLE weighing_acts (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    duration_sec FLOAT NOT NULL,
    left_count INTEGER NOT NULL DEFAULT 0,
    right_count INTEGER NOT NULL DEFAULT 0,
    peak_count INTEGER NOT NULL DEFAULT 0,
    total_weight FLOAT,
    avg_weight FLOAT,
    stream_id VARCHAR(255),
    video_file VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Таблица проходов
CREATE TABLE crossings (
    id BIGSERIAL PRIMARY KEY,
    act_id BIGINT REFERENCES weighing_acts(id) ON DELETE CASCADE,
    pig_id INTEGER NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('left', 'right')),
    crossed_at TIMESTAMPTZ NOT NULL,
    line_x FLOAT NOT NULL,
    line_y FLOAT NOT NULL,
    weight_estimate FLOAT,
    stream_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Таблица схем Excel (для хранения схемы шаблона)
CREATE TABLE excel_schemas (
    id BIGSERIAL PRIMARY KEY,
    template_name VARCHAR(255) NOT NULL,
    schema_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для быстрого поиска
CREATE INDEX idx_weighing_acts_started_at ON weighing_acts(started_at);
CREATE INDEX idx_weighing_acts_video_file ON weighing_acts(video_file);
CREATE INDEX idx_crossings_act_id ON crossings(act_id);
CREATE INDEX idx_crossings_crossed_at ON crossings(crossed_at);

-- Включаем Row Level Security (RLS)
ALTER TABLE weighing_acts ENABLE ROW LEVEL SECURITY;
ALTER TABLE crossings ENABLE ROW LEVEL SECURITY;
ALTER TABLE excel_schemas ENABLE ROW LEVEL SECURITY;

-- Политики доступа (разрешаем все операции для простоты)
CREATE POLICY "Allow all operations on weighing_acts" ON weighing_acts FOR ALL USING (true);
CREATE POLICY "Allow all operations on crossings" ON crossings FOR ALL USING (true);
CREATE POLICY "Allow all operations on excel_schemas" ON excel_schemas FOR ALL USING (true);

-- Создаем пользователей для Supabase
CREATE USER supabase_auth_admin WITH PASSWORD 'root';
CREATE USER supabase_storage_admin WITH PASSWORD 'root';
CREATE USER supabase_admin WITH PASSWORD 'root';
CREATE USER authenticator WITH PASSWORD 'root';

-- Роли
CREATE ROLE anon;
CREATE ROLE authenticated;
CREATE ROLE service_role;

-- Права доступа
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated, service_role;

-- Права для пользователей
GRANT anon TO authenticator;
GRANT authenticated TO authenticator;
GRANT service_role TO authenticator;

GRANT ALL PRIVILEGES ON DATABASE postgres TO supabase_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO supabase_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO supabase_admin;