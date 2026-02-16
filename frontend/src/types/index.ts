export interface Property {
    id: number;
    title: string;
    district: string;
    neighborhood: string;
    postal_code: string;
    latitude: number;
    longitude: number;

    property_type?: string;
    subtype?: string;
    size: number;
    bedrooms: number;
    bathrooms: number;
    floor?: string;
    status?: string;
    new_construction: number;

    price: number;
    price_m2: number;

    lift: number;
    garage: number;
    storage: number;
    terrace: number;
    air_conditioning: number;
    swimming_pool: number;
    garden: number;
    sports: number;

    ingreso?: number;

    vi?: number;
    vo?: number;
    comprable: number;
    roi: number;

    ck: number;
    cm: number;

    // Computed / API enriched
    gap?: number;
    effective_price?: number;
}

export interface GeoJSONFeature {
    type: "Feature";
    geometry: {
        type: "Point";
        coordinates: [number, number];
    };
    properties: Partial<Property>;
}

export interface FeatureCollection {
    type: "FeatureCollection";
    features: GeoJSONFeature[];
}

export interface CatalogItem {
    neighborhood?: string;
    district?: string;
    count: number;
}
