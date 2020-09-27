# Enter your code here. Read input from STDIN. Print output to STDOUT



class FlightTicket:
    def __init__(self,name,org,dest,b_details,t_class,price):
        self.name = name
        self.org = org
        self.dest = dest
        self.b_details = b_details
        self.t_class = t_class
        self.price =  price


class AirlineManagement:
    def __init__(self,airname,passenger):
        self.airname = airname
        self.passenger = passenger


    def findSeatNoWithRowFromBoardingDetails(self):
        nums = ['0','1','2','3','4','5','6','7','8','9']
        alpha = []
        for i in range(65,91):
            alpha.append(chr(i))
      
        li = []
        for pas in self.passenger:
            s = self.passenger[pas].b_details.split(",")
            seat = s[3]
            seat = list(seat)
      
            if seat[0] in nums and seat[1] in nums and seat[2] in alpha:
                pa = []
                pa.append(self.passenger[pas].name)
                #pa.append(" ")
                pa.append(s[3])
                li.append(pa)
                print("li",li)
        
        return li

    def countFlightTicketsOfGivenTravelClass(self,user_class):
        count = 0
        for pas in self.passenger:
            cl = self.passenger[pas].t_class.lower()
            ca = cl.split("_")
            clas = ca[0]
            
            
            user_class = user_class.lower()
            user_class = user_class.split("_")
            user_class = user_class[0]
           
            print("class",clas)
            if  user_class == clas:
                #print("class",user_class,clas)
                count = count + 1

        return count


if __name__ == "__main__":
    n = int(input())
    flight = {}
    for i in range(n):
        name = input()
        org = input()
        dest  = input()
        b_details = input()
        t_class = input()
        price  = float(input())
        f = FlightTicket(name,org,dest,b_details,t_class,price)
        intger = str(i)
        flight.update({intger:f})


    air_class  = input()
    air = AirlineManagement("KPS_Travels",flight)
    tup = air.findSeatNoWithRowFromBoardingDetails()
    if len(tup) != 0:
        for i in range(len(tup)):
            print(tup[i],"\n")

    else:
        print("No appropriate details found")

    c = air.countFlightTicketsOfGivenTravelClass(air_class)
    if c != 0:
        print("Count:",c)

    else:
        print("Count:0")


    


            



























