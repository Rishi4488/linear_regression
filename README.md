
# Ground Cricket Chirps

In The Song of Insects (1948) by George W. Pierce, Pierce mechanically measured the frequency (the number of wing vibrations per second) of chirps (or pulses of sound) made by a striped ground cricket, at various ground temperatures. Since crickets are ectotherms (cold-blooded), the rate of their physiological processes and their overall metabolism are influenced by temperature. Consequently, there is reason to believe that temperature would have a profound effect on aspects of their behavior, such as chirp frequency.
In general, it was found that crickets did not sing at temperatures colder than 60° F or warmer than 100° F.


### Tasks
1. Find the linear regression equation for this data.
2. Chart the original data and the equation on the chart.
3. Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)
4. Extrapolate data: If the ground temperature reached 95° F, then at what approximate rate would you expect the crickets to be chirping?
5. Interpolate data: With a listening device, you discovered that on a particular morning the crickets were chirping at a rate of 18 chirps per second. What was the approximate ground temperature that morning?

# Brain vs. Body Weight
In the file brain_body.txt, the average brain and body weight for a number of mammal species are recorded. Load this data into a Pandas data frame.

### Tasks
1. Find the linear regression equation for this data for brain weight to body weight.
2. Chart the original data and the equation on the chart.
3. Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)

# Salary Discrimination
The file salary.txt contains data for 52 tenure-track professors at a small Midwestern college. This data was used in legal proceedings in the 1980s about discrimination against women in salary.

###The data in the file, by column:
- Sex. 1 for female, 0 for male.
- Rank. 1 for assistant professor, 2 for associate professor, 3 for full professor.
- Year. Number of years in current rank.
- Degree. Highest degree. 1 for doctorate, 0 for master's.
- YSdeg. Years since highest degree was earned.
- Salary. Salary/year in dollars.

### Tasks
1. Find the linear regression equation for this data using columns 1-5 to column 6.
2. Find the selection of columns with the best R2 score.
3. Report whether sex is a factor in salary. Support your argument with graph(s) if appropriate.
