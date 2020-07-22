WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM _all_interaction6_item 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM rand_train_set 
	) 
	-- AND rank >= 6 
    -- AND rank < 20 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
ORDER BY reviewerID,unixReviewTime ASC 
;
WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM _all_interaction6_item 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM rand_train_set 
	) 
	-- AND rank >= 6 
    -- AND rank < 20 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
ORDER BY reviewerID,unixReviewTime ASC 
;


-- OLD
WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM _all_interaction6_item 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM rand_train_set 
	) 
	-- AND rank >= 6 
    -- AND rank < 20 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
ORDER BY reviewerID,unixReviewTime ASC 
;
-- Equal to `SELECT * FROM clothing_userbase_42_oringinal`
WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	ORDER BY RAND() 
	LIMIT 12800 
	) 
, validation_set AS ( 
	SELECT DISTINCT(`asin`) 
	FROM clothing_interaction6_itembase 
	WHERE `asin` NOT IN ( 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 1600 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM validation_set 
	) 
	AND rank = 6 
	ORDER BY `asin`,rank ASC 
)
, candidate_table AS ( 
	SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
	clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
	clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime 
	FROM  clothing_review 
	WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
	ORDER BY reviewerID,unixReviewTime ASC 
)
SELECT * FROM candidate_table 
WHERE rank<6
ORDER BY reviewerID,unixReviewTime ASC 
;