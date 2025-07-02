-- 16-glam_rock.sql
-- Lists all bands with Glam rock as their main style, ranked by longevity
SELECT band_name, (IFNULL(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style = 'Glam rock'
ORDER BY lifespan DESC; 