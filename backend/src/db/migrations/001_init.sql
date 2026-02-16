CREATE EXTENSION IF NOT EXISTS postgis;

-- Usuarios
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Propiedades (subset "core" para la UI + ranking)
CREATE TABLE IF NOT EXISTS properties (
    id BIGINT PRIMARY KEY,
    title TEXT,
    district TEXT,
    neighborhood TEXT,
    postal_code TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom geometry(Point, 4326),

    -- Descripción
    property_type TEXT,
    subtype TEXT,
    size NUMERIC,
    bedrooms INT,
    bathrooms INT,
    floor TEXT,
    status TEXT,
    new_construction INT,

    -- Precio
    price NUMERIC,
    price_m2 NUMERIC,

    -- Amenities (0/1)
    lift INT,
    garage INT,
    storage INT,
    terrace INT,
    air_conditioning INT,
    swimming_pool INT,
    garden INT,
    sports INT,

    -- Contexto
    ingreso NUMERIC,

    -- Clasificación / inversión
    vi NUMERIC,
    vo NUMERIC,
    comprable INT,
    roi NUMERIC,

    -- Probabilidades (2 meses)
    ck NUMERIC,  -- propietario (prob_propietario_1)
    cm NUMERIC,  -- inversor (prob_investor_1)

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Índices geoespaciales y otros
CREATE INDEX IF NOT EXISTS idx_properties_geom ON properties USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_properties_neighborhood ON properties(neighborhood);
CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price);
CREATE INDEX IF NOT EXISTS idx_properties_roi ON properties(roi);

-- Favoritos
CREATE TABLE IF NOT EXISTS favorites (
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    property_id BIGINT NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, property_id)
);
