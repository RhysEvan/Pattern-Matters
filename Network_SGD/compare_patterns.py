from Network_SGD.training_UNet import unet_train

# pattern_list = ["lines-saw", "dots", "checkers", "grating", "flowers", "honey"]
pattern_list = ["lines-saw"]

# Loop through patterns
for pattern in pattern_list:
    
    # periods_list = [6]
    periods_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24]

    # Loop through number of periods
    for n_periods in periods_list:

        name = f"{pattern}_{n_periods}p"
        
        start_set = 1
        end_set = 4

        # Loop through sets
        for idx in range(start_set, end_set+1):

            unet_train(name, idx)
