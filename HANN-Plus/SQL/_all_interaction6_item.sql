-- batch 64 RGM
SELECT * FROM _all_interaction6_item 
WHERE _turn = 1
ORDER BY _turn, `asin`, rank
LIMIT 106752
;
SELECT * FROM _all_interaction6_item 
WHERE _turn = 2
AND _no> 107460
AND _no <= 133188
-- +25728
ORDER BY _turn, `asin`, rank
;

-- batch 64 (Full train PRM:1.033, RGM:repeat) 0702
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
LIMIT 580992
;
SELECT * FROM _all_interaction6_item 
WHERE _no> 580992
AND _no <= 653568
ORDER BY _turn, `asin`, rank
;

-- batch 80 RGM
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
LIMIT 580800
;
SELECT * FROM _all_interaction6_item 
WHERE _no> 580800
AND _no <= 653280
ORDER BY _turn, `asin`, rank
;

-- 0701
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
LIMIT 581142
;
SELECT * FROM _all_interaction6_item 
WHERE _no> 581142
AND _no <= 593250
ORDER BY _turn, `asin`, rank
;


-- OLD
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
;
WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	LIMIT 12800 
	) 
, validate_set AS (
	SELECT DISTINCT(`asin`)
	FROM clothing_interaction6_itembase
	WHERE `asin` NOT IN (
		SELECT * FROM rand_train_set 
		) 
	LIMIT 1600 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM validate_set 
) 
AND clothing_interaction6_itembase.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC 
;