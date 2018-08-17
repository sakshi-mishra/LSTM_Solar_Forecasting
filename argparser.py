import argparse

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('l',
                        help="Location whose data needs to be trained/tested with"+
                        "Values can be one of [Bondville, Boulder, Desert_Rock,"
                                            + "Fort_Peck,Goodwin_Creek, Penn_State,"
                                             + "Sioux_Falls]")
    parser.add_argument('y',
                        help='4 digit Test year. One among [2009,2015,2016,2017]')
    parser.add_argument('t',
                        help='True or False.To train using 2010-2011 data or not')
    parser.add_argument('--num-epochs', default = 1000, type=int,
                        help="Number of training, testing epochs")
    args, _ = parser.parse_known_args()

    # Sanity check the arguments
    # args.y
    test_year = args.y
    if test_year not in ["2009", "2015", "2016", "2017"]:
        print("Test year argument is not valid. Exiting...")
        parser.print_help()
        exit()
    # args.t
    if args.t in ["True", "true"]:
        run_train = True
    elif args.t in ["False", "false"]:
        run_train = False
    else:
        print("Train flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()
    # args.l
    test_location = args.l
    if test_location not in ["Bondville", "Boulder", "Desert_Rock",
                             "Fort_Peck,Goodwin_Creek, Penn_State", "Sioux_Falls"]:
        print("Test location is not valid.Exiting...")
        parser.print_help()
        exit()
    # args.num_epochs
    num_epochs = args.num_epochs

    print("test_location=",test_location, "test_year=",test_year,"run_train=",
          run_train, "num_epochs=", num_epochs)

    return test_location, test_year, run_train, num_epochs
