WITH _title AS ( 
	SELECT clothing_review.`asin`, clothing_metadata.title 
	FROM clothing_metadata, clothing_review 
	WHERE clothing_metadata.`asin` = clothing_review.`asin` 
	GROUP BY clothing_review.`asin` 
) 
SELECT * 
FROM _title;