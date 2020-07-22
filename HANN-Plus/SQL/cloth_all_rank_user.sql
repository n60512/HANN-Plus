SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_interaction6_rm_sw.reviewText, clothing_review.unixReviewTime 
FROM clothing_review , clothing_interaction6_rm_sw 
WHERE clothing_review.ID = clothing_interaction6_rm_sw.ID 
ORDER BY reviewerID,unixReviewTime ASC ;