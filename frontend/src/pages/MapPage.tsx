import { useState } from 'react';
import Map from '../components/Map';
import PropertyDrawer from '../components/PropertyDrawer';
import { useProperties } from '../hooks/useProperties';
import type { Property } from '../types';
import { client } from '../api/client';
import { useFilters } from '../contexts/FilterContext';

export default function MapPage() {
    const [bbox, setBbox] = useState<string>('');
    const [propertyDetails, setPropertyDetails] = useState<Property | null>(null);
    const [isDrawerOpen, setIsDrawerOpen] = useState(false);

    const { filters } = useFilters();
    const { data: properties, loading, error } = useProperties(bbox, filters);

    const handleSelectProperty = async (id: number) => {
        setPropertyDetails(null);
        setIsDrawerOpen(true);

        try {
            const res = await client.get<Property>(`/properties/${id}`);
            if (res.data) {
                setPropertyDetails(res.data);
            }
        } catch (err) {
            console.error("Failed to fetch property details", err);
        }
    };

    const handleCloseDrawer = () => {
        setIsDrawerOpen(false);
        setPropertyDetails(null);
    };

    return (
        <div className="relative w-full h-full">
            {loading && (
                <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-white px-4 py-2 rounded-full shadow-md z-[400] text-sm font-semibold text-blue-600">
                    Cargando...
                </div>
            )}

            {error && (
                <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-100 px-4 py-2 rounded-full shadow-md z-[400] text-sm font-semibold text-red-600">
                    {error}
                </div>
            )}

            <Map
                properties={properties}
                onBoundsChange={setBbox}
                onSelectProperty={handleSelectProperty}
                city={filters.city}
            />

            <PropertyDrawer
                property={propertyDetails}
                isOpen={isDrawerOpen}
                onClose={handleCloseDrawer}
            />
        </div>
    );
}
