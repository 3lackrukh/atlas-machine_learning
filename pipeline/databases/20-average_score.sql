-- 20-average_score.sql
-- Creates a stored procedure ComputeAverageScoreForUser to compute and store the average score for a student
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    DECLARE avg_score DECIMAL(5,2);
    -- Calculate the average score for the user
    SELECT AVG(score) INTO avg_score FROM corrections WHERE corrections.user_id = user_id;
    -- Update the user's average_score
    UPDATE users SET average_score = avg_score WHERE id = user_id;
END;//