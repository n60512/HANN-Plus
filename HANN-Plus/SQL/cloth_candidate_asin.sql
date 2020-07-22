WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	ORDER BY RAND() 
	LIMIT 15040 	
	-- LIMIT 15000 
	-- LIMIT 1000 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM rand_train_set 
	) 
	AND rank = 6 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_interaction6_rm_sw.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review , clothing_interaction6_rm_sw 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
AND clothing_review.ID = clothing_interaction6_rm_sw.ID 
ORDER BY reviewerID,unixReviewTime ASC 
;
WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	ORDER BY RAND() 
	LIMIT 15040 	
	-- LIMIT 15000 
	-- LIMIT 1000 
	) 
, tmptable AS ( 
	SELECT DISTINCT(`asin`) 
	FROM clothing_interaction6_itembase 
	WHERE `asin` NOT IN ( 
		-- SELECT * FROM clothing_interaction6_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 2000 
	-- LIMIT 200 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM tmptable 
	) 
	AND rank = 6 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_interaction6_rm_sw.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review , clothing_interaction6_rm_sw 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
AND clothing_review.ID = clothing_interaction6_rm_sw.ID 
ORDER BY reviewerID,unixReviewTime ASC ;
;