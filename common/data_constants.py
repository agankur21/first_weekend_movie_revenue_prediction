TRAINING_START_YEAR=2007
TRAINING_END_YEAR=2013
TEST_START_YEAR=2014
TEST_END_YEAR=2016
DAYS_BEFORE_RELEASE=30

def get_dates_before(given_date,number_of_days_prior):
    pass


def get_zero_padded_number_string(number,length):
    given_length=len(str(number))
    if given_length > length:
        return str(number)[(given_length-length):]
    else:
        return '0'*(length-given_length) + str(number)



if __name__ == '__main__':
    #Unit Testing functions
    print get_zero_padded_number_string(1,3)
    print get_zero_padded_number_string(10, 3)
    print get_zero_padded_number_string(100, 3)
    print get_zero_padded_number_string(1002, 3)