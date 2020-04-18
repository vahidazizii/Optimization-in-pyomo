from __future__ import print_function

import math
import cplex
import pandas as pd
import xlsxwriter
import time

# region Creating Excel file for recording results
# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('Benders.xlsx')
worksheet = workbook.add_worksheet('ObjTime')
# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 20)
# Write some numbers, with row/column notation.
worksheet.write(0, 0, "Instance")
worksheet.write(0, 1, "Objective Value")
worksheet.write(0, 2, "Time")
# endregion

for cc in range(1,31):
    print(cc)
    filename=str(cc)+".xlsx"
    print(filename)


    # region DEFINING PARAMETERS
    num_sup = pd.read_excel(filename, sheetname='supp_num', header=None)  # number of suppliers
    num_cr= pd.read_excel(filename, sheetname='cross_num', header=None)       #number of crossdocks
    num_del = pd.read_excel(filename, sheetname='del_num', header=None)  # number of delivery customers
    num_prod = pd.read_excel(filename, sheetname='prod_num', header=None)  # number of products
    num_del_v = pd.read_excel(filename, sheetname='num_del_v', header=None)  # number of delivery vehicles
    num_sup_v = pd.read_excel(filename, sheetname='num_sup_v', header=None)  # number of pickup vehicles
    cap_v = pd.read_excel(filename, sheetname='cap_v', header=None)  # capacity of delivery vehicles
    cap_vp = pd.read_excel(filename, sheetname='cap_vp', header=None)  # capacity of pickup vehicles
    FV_p = pd.read_excel(filename, sheetname='FV_p', header=None)  # fixed cost of pickup vehicles
    FV_d = pd.read_excel(filename, sheetname='FV_d', header=None)  # fixed cost of delivery vehicles
    FV_Dr = pd.read_excel(filename, sheetname='FV_Dr', header=None)  # fixed cost of direst shipment vehicle
    cap_c = pd.read_excel(filename, sheetname='cap_c', header=None)  # Capacity of crossdock c
    cap_s = pd.read_excel(filename, sheetname='cap_s', header=None)  # Maximum amount of supply by supplier p
    cdr = pd.read_excel(filename, sheetname='cdr',header=None)  # cost of direct supply from supplier su to customer cu >>cost of arcs between suppliers and customers
    cd = pd.read_excel(filename, sheetname='cd', header=None)  # distance between node i to node j >> distances among customers and crossdocks
    FW = pd.read_excel(filename, sheetname='FW', header=None)  # Openning cost of crossdock c
    dem = pd.read_excel(filename, sheetname='dem', header=None)  # demand of customer
    cp = pd.read_excel(filename, sheetname='cp',header=None)  # distance between node p to node t >>distances among suppliers and crossdocks
    b = pd.read_excel(filename, sheetname='b', header=None)  # identifier of type of demanded good(r) by customer j
    a = pd.read_excel(filename, sheetname='a',header=None)  # identifier of type of produced good(r) by supplier p

    num_cr=[num_cr.values[0][0]]
    num_del=[num_del.values[0][0]]
    num_sup=[num_sup.values[0][0]]
    num_prod=[num_prod.values[0][0]]
    num_del_v=[num_del_v.values[0][0]]
    num_sup_v=[num_sup_v.values[0][0]]
    cap_v=[cap_v.values[0][0]]
    cap_vp=[cap_vp.values[0][0]]
    FV_Dr=[FV_Dr.values[0][0]]
    FV_p=[FV_p.values[0][0]]
    FV_d=[FV_d.values[0][0]]
    # endregion

    # region DEFINING SETS

    cros_set = list(range(1, num_cr[0] + 1))
    del_set = list(range(num_cr[0] + 1, num_cr[0] + num_del[0] + 1))
    sup_set = list(range(num_cr[0] + num_del[0] + 1, num_cr[0] + num_del[0] + num_sup[0] + 1))
    cros_del_set = list(range(1, num_cr[0] + num_del[0] + 1))
    cros_sup_set = list(range(1, num_cr[0] + 1))
    for i in sup_set:
        cros_sup_set.append(i)

    all_nodes_set = list(range(1, num_cr[0] + num_del[0] + 1))
    for i in sup_set:
        all_nodes_set.append(i)

    prod_set = list(range(1, num_prod[0] + 1))
    # endregion

    # region DEFINING VARIABLES
    x = []  # arc between nodes i and j by vehicle v
    w = []  # Opening crossdock
    z = []  # assignment of customer i to crossdock j
    u = []  # load on vehicle v during travling in arc (i,j)
    D = []  # Diret shipment from supplier p to customer i
    f = []  # objctive function value
    y = []  # arc between pickp node t and p by vehicle v originated from crossdock i
    s = []  # load of vehicle v, originated from crossdock i, during arc t-p
    g = []  # amount of pickup of product r from node p originated from crossdock i


    # endregion

    # Defining problem
    def setupproblem():
        problem = cplex.Cplex()
        problem.objective.set_sense(problem.objective.sense.minimize)

        # region ADDING VARIABLES

        # x(i,j,v) arc between nodes i and j by vehicle v
        allxvars = []
        upperbound = []
        xcoef = []
        for i in cros_del_set:

            x.append([])
            for j in cros_del_set:
                x[i - 1].append([])

                for v in range(1, num_del_v[0] + 1):
                    varname = "x(" + str(i) + "," + str(j) + "," + str(v) + ")"
                    allxvars.append(varname)
                    x[i - 1][j - 1].append(varname)
                    if i <= num_cr[0] and j > num_cr[0]:
                        xcoef.append(float(FV_d[0]))
                    else:
                        xcoef.append(0)

                    same_product = 1
                    if i > num_cr[0] and j > num_cr[0]:
                        same_product = 0
                        for r in range(len(prod_set)):

                            same_product += float(float(b.values[i - len(cros_set) - 1][r])) * float(float(b.values[j - len(cros_set) - 1][r]))

                    if (i < (float(num_cr[0]) + 1) and j < (float(num_cr[0]) + 1)) or i == j or same_product == 0:
                        upperbound.append(0)
                    else:
                        upperbound.append(1)

        xSET = list(problem.variables.add(names=allxvars, lb=[0] * len(allxvars),
                                          ub=upperbound,
                                          types=["B"] * len(allxvars), obj=xcoef))

        # w(j) Opening crossdock j
        allwvars = []
        wcoef = []
        for j in cros_set:
            varname = "w(" + str(j) + ")"
            allwvars.append(varname)
            w.append(varname)

            wcoef.append(float(FW.values[j - 1]))

        wSET = list(problem.variables.add(names=allwvars, lb=[0] * len(allwvars),
                                          ub=[1] * len(allwvars),
                                          types=["B"] * len(allwvars), obj=wcoef))

        # z(i,j) assignment of customer i to crossdock j
        allzvars = []

        for i in range(len(del_set)):
            z.append([])
            for j in range(len(cros_set)):
                varname = "z(" + str(del_set[i]) + "," + str(cros_set[j]) + ")"
                allzvars.append(varname)
                z[i].append(varname)

        zSET = list(problem.variables.add(names=allzvars, lb=[0] * len(allzvars),
                                          ub=[1] * len(allzvars),
                                          types=["B"] * len(allzvars)))

        # u(i,j,v) load on vehicle v during travling in arc (i,j)
        alluvars = []
        upperbound = []
        ucoef = []
        for i in cros_del_set:
            u.append([])
            for j in cros_del_set:
                u[i - 1].append([])
                for v in range(1, num_del_v[0] + 1):
                    varname = "u(" + str(i) + "," + str(j) + "," + str(v) + ")"
                    alluvars.append(varname)
                    u[i - 1][j - 1].append(varname)\

                    ucoef.append(float(cd.values[i - 1][j - 1]))

                    if i < (float(num_cr[0]) + 1) and j < (float(num_cr[0]) + 1):
                        upperbound.append(0)
                    else:
                        upperbound.append(float(cap_v[0]))
        uSET = list(problem.variables.add(names=alluvars, lb=[0] * len(alluvars),
                                          ub=upperbound,
                                          types=["C"] * len(alluvars), obj=ucoef))

        # D(p,i) Diret shipment from supplier p to customer i
        allDvars = []
        Dcoef = []
        for i in range(len(sup_set)):
            D.append([])
            for j in range(len(del_set)):
                varname = "D(" + str(sup_set[i]) + "," + str(del_set[j]) + ")"
                allDvars.append(varname)
                D[i].append(varname)
                Dcoef.append(float(FV_Dr[0])+(float(cdr.values[i][j])*float(dem.values[j])))

        DSET = list(problem.variables.add(names=allDvars, lb=[0] * len(allDvars),
                                          ub=[1] * len(allDvars),
                                          types=["B"] * len(allDvars), obj=Dcoef))

        # y(i,t,p,vp) arc between pickp node t and p by vehicle v originated from crossdock i

        upper_bound = []
        allyvars = []
        ycoef = []
        for i in range(len(cros_set)):
            y.append([])
            for t in range(len(cros_sup_set)):
                y[i].append([])

                for p in range(len(cros_sup_set)):
                    y[i][t].append([])

                    for v in range(1, num_sup_v[0] + 1):
                        varname = "y(" + str(cros_set[i]) + "," + str(cros_sup_set[t]) + "," + str(
                            cros_sup_set[p]) + "," + str(v) + ")"
                        allyvars.append(varname)
                        y[i][t][p].append(varname)

                        if t == i and p >= 3:

                            ycoef.append(float(FV_p[0]))
                        else:
                            ycoef.append(1)

                        if (t < len(cros_set) and p < len(cros_set)) or (
                                i < len(cros_set) and t < len(cros_set) and i != t):
                            upper_bound.append(0)

                        else:
                            upper_bound.append(1)

        ySET = list(problem.variables.add(names=allyvars, lb=[0] * len(allyvars),
                                          ub=upper_bound,
                                          types=["B"] * len(allyvars), obj=ycoef))

        # s(i,t,p,vp) load of vehicle v, originated from crossdock i, during arc t-p

        allsvars = []
        scoefs = []
        for i in range(len(cros_set)):
            s.append([])
            for t in range(len(cros_sup_set)):
                s[i].append([])

                for p in range(len(cros_sup_set)):
                    s[i][t].append([])

                    for v in range(1, num_sup_v[0] + 1):
                        varname = "s(" + str(cros_set[i]) + "," + str(cros_sup_set[t]) + "," + str(
                            cros_sup_set[p]) + "," + str(v) + ")"
                        allsvars.append(varname)
                        s[i][t][p].append(varname)
                        scoefs.append(float(cp.values[t][p]))

        sSET = list(problem.variables.add(names=allsvars, lb=[0] * len(allsvars),
                                          ub=[float(cap_vp[0])] * len(allsvars),
                                          types=["C"] * len(allsvars), obj=scoefs))

        # g(i,p,r,vp) amount of pickup of product r from node p originated from crossdock i

        allgvars = []
        ub = []

        for i in range(len(cros_set)):
            g.append([])

            for p in range(len(cros_sup_set)):
                g[i].append([])

                for r in range(len(prod_set)):
                    g[i][p].append([])

                    for v in range(1, num_sup_v[0] + 1):
                        varname = "g(" + str(cros_set[i]) + "," + str(cros_sup_set[p]) + "," + str(prod_set[r]) + "," + str(
                            v) + ")"
                        allgvars.append(varname)
                        if p < num_cr[0]:
                            ub.append(0)
                        else:
                            ub.append(float(cap_s.values[r][0]))
                        g[i][p][r].append(varname)

        gSET = list(problem.variables.add(names=allgvars, lb=[0] * len(allgvars),
                                          ub=ub,
                                          types=["C"] * len(allgvars)))
        # endregion

        # region ADDING constraints

        # CrDockLocation(i,j)$((ord(i)>nc) and (ord(j)<nc+1))..z(i,j)=l=w(j);

        for i in range(len(del_set)):
            for j in range(len(cros_set)):
                zvars = []
                zvars.append(z[i][j])
                wvars = []
                wvars.append(w[j])
                allvars = zvars + wvars
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(allvars, list([1] * len(zvars)) + list([-1] * len(wvars)))],
                    senses=["L"],
                    rhs=[0])

        # capacity_crossdock(j)$(ord(j) < nc + 1)..sum((i)$(ord(i) > nc), dem(i) * z(i, j))=l = cap_c(j);

        for j in range(len(cros_set)):
            zvars = []
            coef = []
            for i in range(len(del_set)):
                zvars.append(z[i][j])
                coef.append(float(dem.values[i]))
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(zvars, coef)],
                senses=["L"],
                rhs=[float(cap_c.values[j])])

        # routing1(i,v)$(ord(i)<nc+1)..sum((j)$(ord(j)>nc),x(i,j,v))=l=1;
        for i in cros_set:
            for v in range(num_del_v[0]):
                thevars = []
                for j in del_set:
                    thevars.append(x[i - 1][j - 1][v])
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(thevars, [1] * len(thevars))],
                    senses=["L"],
                    rhs=[1])

        # routing2(i,j,v)$((ord(j)<nc+1) and (ord(i)>nc))..x(i,j,v)=l=z(i,j);
        for i in range(len(del_set)):
            for j in range(len(cros_set)):
                for v in range(num_del_v[0]):
                    zvars = []
                    zcoef = []
                    zvars.append(z[i][j])
                    zcoef.append(-1)

                    xvars = []
                    xcoef = []

                    xvars.append(x[del_set[i] - 1][j][v])
                    xcoef.append(1)

                    thevars = xvars + zvars
                    thecoefs = xcoef + zcoef

                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"],
                        rhs=[0])

        # routing3(i,j,v)$((ord(j)<nc+1) and (ord(i)>nc))..x(j,i,v)=l=z(i,j);
        for i in range(len(del_set)):
            for j in range(len(cros_set)):
                for v in range(num_del_v[0]):
                    zvars = []
                    zcoef = []
                    zvars.append(z[i][j])
                    zcoef.append(-1)

                    xvars = []
                    xcoef = []

                    xvars.append(x[j][del_set[i] - 1][v])
                    xcoef.append(1)

                    thevars = xvars + zvars
                    thecoefs = xcoef + zcoef

                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"],
                        rhs=[0])

        # routing4(i)$(ord(i) > nc)..sum((v, j)$(ord(i) <> ord(j)), x(i, j, v))=e = sum((j)$(ord(j) < nc + 1), z(i, j));

        for i in range(len(del_set)):
            zvars = []
            zcoef = []
            xvars = []
            xcoef = []
            for j in range(len(cros_del_set)):
                if j < len(cros_set):
                    zvars.append(z[i][j])
                    zcoef.append(-1)
                for v in range(num_del_v[0]):
                    xvars.append(x[del_set[i] - 1][j][v])
                    xcoef.append(1)

            thevars = xvars + zvars
            thecoefs = xcoef + zcoef
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["L"],
                rhs=[0], names=["routing4_{0}".format(i + 1)])

        # routing5(i,v)..sum((l)$(ord(l)<>ord(i)),x(l,i,v))-sum((j)$(ord(i)<>ord(j)),x(i,j,v))=e=0;

        for i in cros_del_set:
            for v in range(num_del_v[0]):
                thevars1 = []
                thevars2 = []
                coef1 = []
                coef2 = []
                for j in cros_del_set:
                    if i != j:
                        thevars1.append(x[i - 1][j - 1][v])
                        thevars2.append(x[j - 1][i - 1][v])
                        coef1.append(-1)
                        coef2.append(1)
                        thevars = thevars2 + thevars1
                        coef = coef2 + coef1
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(thevars, coef)],
                    senses=["E"],
                    rhs=[0])

        # routing7(i,j,v,l)$((ord(i)>nc) and (ord(j)>nc) and (ord(l)<nc+1))..x(i,j,v)+z(i,l)+sum(jj $((ord(jj)<nc+1)and (ord(jj)<>ord(l))),z(j,jj))=l=2;
        for i in range(len(del_set)):
            for j in range(len(del_set)):
                for l in range(len(cros_set)):
                    for v in range(num_del_v[0]):

                        zvars = []
                        zcoef = []
                        zvars.append(z[i][l])
                        zcoef.append(1)

                        xvars = []
                        xcoef = []
                        xvars.append(x[del_set[i] - 1][del_set[j] - 1][v])
                        xcoef.append(1)

                        zvars2 = []
                        zcoef2 = []
                        for jj in range(len(cros_set)):
                            if jj != l:
                                zvars2.append(z[j][jj])
                                zcoef2.append(1)

                        thevars = xvars + zvars + zvars2
                        thecoefs = xcoef + zcoef + zcoef2
                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                            senses=["L"],
                            rhs=[2])

        # routing8(i)$(ord(i)>nc)..sum((j,v),u(j,i,v))-sum((j,v),u(i,j,v))=e=dem(i)*sum((j)$(ord(j)<nc+1),z(i,j));

        for i in range(len(del_set)):
            zvars = []
            zcoef = []
            uvars = []
            ucoef = []
            uvars2 = []
            ucoef2 = []
            for j in range(len(cros_del_set)):
                if cros_del_set[j] - 1 != del_set[i] - 1:
                    if j < (len(cros_set)):
                        zvars.append(z[i][j])
                        zcoef.append(-float(dem.values[i]))
                    for v in range(num_del_v[0]):
                        uvars.append(u[cros_del_set[j] - 1][del_set[i] - 1][v])
                        ucoef.append(1)
                        uvars2.append(u[del_set[i] - 1][cros_del_set[j] - 1][v])
                        ucoef2.append(-1)

            thevars = uvars + uvars2 + zvars
            thecoefs = ucoef + ucoef2 + zcoef
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["E"],
                rhs=[0])

        #     routing9(j)$(ord(j)<nc+1)..sum((i,v)$(ord(i)>nc),u(j,i,v))=e=sum((i)$(ord(i)>nc),z(i,j)*dem(i));
        for j in range(len(cros_set)):
            zvars = []
            zcoef = []
            uvars = []
            ucoef = []
            for i in range(len(del_set)):

                if j < (len(cros_set)):
                    zvars.append(z[i][j])
                    zcoef.append(-float(dem.values[i]))

                for v in range(num_del_v[0]):
                    uvars.append(u[j][del_set[i] - 1][v])
                    ucoef.append(1)

            thevars = uvars + zvars
            thecoefs = ucoef + zcoef
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["E"],
                rhs=[0])


        # routing11(i,j,v)..u(i,j,v)=l=(cap_v-dem(i))*x(i,j,v);
        for i in range(len(cros_del_set)):
            for j in range(len(cros_del_set)):
                for v in range(num_del_v[0]):
                    uvars = []
                    ucoef = []
                    uvars.append(u[i][j][v])
                    ucoef.append(1)

                    xvars = []
                    xcoef = []
                    xvars.append(x[i][j][v])
                    if i<num_cr[0]:
                        xcoef.append(0 - float(cap_v[0]))
                    else:
                        xcoef.append(float(dem.values[i-num_del[0]]) - float(cap_v[0]))



                    thevars = xvars + uvars
                    thecoefs = xcoef + ucoef
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"],
                        rhs=[0])



        # number_of_delivery_vehicles(i)$(ord(i)<nc+1)..sum((j,v)$(ord(j)>nc),x(i,j,v))=l=ndv;
        for i in range(len(cros_set)):

            xvars = []
            xcoef = []

            for j in range(len(del_set)):
                for v in range(num_del_v[0]):
                    xvars.append(x[i][del_set[j] - 1][v])
                    xcoef.append(1)

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(xvars, xcoef)],
                senses=["L"],
                rhs=[float(num_del_v[0])])

        # pickup_assignment1(i,vp,t,p)$((ord(i)<(nc+1))and (ord(p)>(nc)) and (ord(t)=ord(i))

        for i in cros_set:
            wvars = []
            wcoef = []
            wvars.append(w[i - 1])
            wcoef.append(-1)

            for t in cros_set:
                if t == i:
                    for p in sup_set:
                        for vp in range(num_sup_v[0]):
                            yvars = []
                            ycoef = []
                            yvars.append(y[i - 1][t - 1][p - len(cros_del_set) + len(cros_set) - 1][vp])
                            ycoef.append(1)

                            allvars = yvars + wvars
                            allvarscoef = ycoef + wcoef
                            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                                senses=["L"],
                                rhs=[0])

        # pickup_assignment2(r,i,t)$((ord(i)<nc+1)and (ord(t)=ord(i)) ).. sum((vp,p)$(ord(p)>nc),a(p,r)*y(i,vp,t,p))=l=1;

        for r in prod_set:

            for i in cros_set:
                yvars = []
                ycoef = []
                for t in cros_set:
                    if t == i:

                        for p in sup_set:
                            for vp in range(num_sup_v[0]):
                                yvars.append(y[i - 1][t - 1][p - len(cros_del_set) + len(cros_set) - 1][vp])
                                ycoef.append(float(a.values[p - len(cros_del_set) - 1][r - 1]))

                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(yvars, ycoef)],
                    senses=["L"],
                    rhs=[1])

        # pickup_assignment3(r,i,t,j)$((ord(i)<nc+1)and (ord(t)=ord(i)) and (ord(j)>nc) ).. sum((vp,p)$(ord(p)>nc),a(p,r)*y(i,vp,t,p))=g=b(j,r)*z(j,i);

        for r in prod_set:
            for j in del_set:
                for i in cros_set:
                    yvars = []
                    ycoef = []
                    zvars = []
                    zcoef = []
                    zvars.append(z[j - len(cros_set) - 1][i - 1])
                    zcoef.append(float(b.values[j - len(cros_set) - 1][r - 1]))
                    for t in cros_set:
                        if t == i:

                            for p in sup_set:
                                for vp in range(num_sup_v[0]):
                                    yvars.append(y[i - 1][t - 1][p - len(cros_del_set) + len(cros_set) - 1][vp])
                                    ycoef.append(-float(a.values[p - len(cros_del_set) - 1][r - 1]))

                    allvars = yvars + zvars
                    allvarscoef = ycoef + zcoef
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                        senses=["L"],
                        rhs=[0])


        # pickup_assignment3_3(r,i,t,j)$((ord(i)<nc+1)and (ord(t)=ord(i)) and (ord(j)>nc) ).. sum((vp,p)$(ord(p)>nc),a(p,r)*y(i,vp,t,p))=g=b(j,r)*z(j,i);

        for r in prod_set:
            for j in del_set:
                for i in cros_set:
                    yvars = []
                    ycoef = []
                    zvars = []
                    zcoef = []
                    zvars.append(z[j - len(cros_set) - 1][i - 1])
                    zcoef.append(float(b.values[j - len(cros_set) - 1][r - 1]))
                    for p in cros_set:
                        if p == i:

                            for t in sup_set:
                                for vp in range(num_sup_v[0]):
                                    yvars.append(y[i - 1][t - len(cros_del_set) + len(cros_set) - 1][p - 1][vp])
                                    ycoef.append(-float(a.values[t - len(cros_del_set) - 1][r - 1]))

                    allvars = yvars + zvars
                    allvarscoef = ycoef + zcoef
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                        senses=["L"],
                        rhs=[0])

        # pickup_routing1(i,vp,t)$((ord(i)<(nc+1)) and (ord(t)<(nc+1)) and (ord(t)=ord(i)))..sum((p)$(ord(p)>(nc)),y(i,vp,t,p))=l=1;

        for i in cros_set:

            for t in cros_set:
                if t == i:
                    for vp in range(num_sup_v[0]):
                        yvars = []
                        ycoef = []
                        for p in sup_set:
                            yvars.append(y[i - 1][t - 1][p - len(cros_del_set) + len(cros_set) - 1][vp])
                            ycoef.append(1)

                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(yvars, ycoef)],
                            senses=["L"],
                            rhs=[1])

            # pickup_routing2(i,vp,p)$(ord(i)<(nc+1))..sum((t)$(ord(t)<>ord(p)),y(i,vp,t,p))-sum((t)$(ord(p)<>ord(t)),y(i,vp,p,t))=e=0;

            for i in cros_set:

                for t in range(len(sup_set)):

                    for vp in range(num_sup_v[0]):
                        yvars1 = []
                        ycoef1 = []
                        yvars2 = []
                        ycoef2 = []
                        for p in range(len(cros_sup_set)):
                            if (t + len(cros_set)) != p:
                                yvars1.append(y[i - 1][t + len(cros_set)][p][vp])
                                ycoef1.append(-1)

                                yvars2.append(y[i - 1][p][t + len(cros_set)][vp])
                                ycoef2.append(1)

                                allvars = yvars1 + yvars2
                                allvarscoef = ycoef1 + ycoef2

                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                            senses=["E"],
                            rhs=[0])
        # pickup_routing3(i,vp,t,p)$((ord(i)<(nc+1))and (ord(t)<>ord(p)) and (ord(p)>(nc)) and (ord(t)>(nc)) ).. y(i,vp,t,p)+y(i,vp,p,t)=l=1;

        for i in cros_set:

            for t in sup_set:

                for vp in range(num_sup_v[0]):

                    for p in sup_set:
                        yvars1 = []
                        ycoef1 = []
                        yvars2 = []
                        ycoef2 = []
                        if p != t:
                            yvars1.append(y[i - 1][t - len(cros_del_set) + len(cros_set) - 1][
                                              p - len(cros_del_set) + len(cros_set) - 1][vp])
                            ycoef1.append(1)

                            yvars2.append(y[i - 1][p - len(cros_del_set) + len(cros_set) - 1][
                                              t - len(cros_del_set) + len(cros_set) - 1][vp])
                            ycoef2.append(1)
                            allvars = yvars1 + yvars2
                            allvarscoef = ycoef1 + ycoef2

                            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                                senses=["L"],
                                rhs=[1])

        # pickup_routing4(i,vp,p,t)$(ord(i)<(nc+1) and (ord(p)>(nc)) and (ord(t)>(nc)) and (ord(t)<>ord(p)) )..y(i,vp,p,t)=l=sum(r,a(p,r)*a(t,r));

        for i in cros_set:

            for t in sup_set:

                for vp in range(num_sup_v[0]):

                    for r in prod_set:

                        for p in sup_set:

                            if p != t:

                                yvars = []
                                ycoef = []
                                yvars.append(y[i - 1][p - len(cros_del_set) + len(cros_set) - 1][
                                                 t - len(cros_del_set) + len(cros_set) - 1][vp])
                                ycoef.append(1)

                                rhside = 0
                                for r in prod_set:
                                    rhside += (float(a.values[p - len(cros_del_set) - 1][r - 1]) * float(
                                        a.values[t - len(cros_del_set) - 1][r - 1]))

                                problem.linear_constraints.add(
                                    lin_expr=[cplex.SparsePair(yvars, ycoef)],
                                    senses=["L"],
                                    rhs=[rhside])

        # pickup_load1(i,vp,p,r)$(ord(i)<(nc+1)).. g(i,vp,p,r)=l=sum(t,y(i,vp,t,p)*cap_vp);
        for r in prod_set:
            for i in cros_set:

                for p in range(len(cros_sup_set)):

                    for vp in range(num_sup_v[0]):
                        yvars = []
                        ycoef = []
                        gvars = []
                        gcoef = []
                        gvars.append(g[i - 1][p][r - 1][vp])
                        gcoef.append(1)
                        for t in range(len(cros_sup_set)):
                            if p != t:
                                yvars.append(y[i - 1][t][p][vp])
                                ycoef.append(-float(cap_vp[0]))

                                allvars = yvars + gvars
                                allvarscoef = ycoef + gcoef

                        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                            senses=["L"],
                            rhs=[0])

        # pickup_load2(i,vp,r)$(ord(i)<(nc+1)).. sum(p,g(i,vp,p,r))=l=cap_vp;

        for r in prod_set:
            for i in cros_set:
                for vp in range(num_sup_v[0]):

                    gvars = []
                    gcoef = []
                    for p in range(len(cros_sup_set)):
                        gvars.append(g[i - 1][p][r - 1][vp])
                        gcoef.append(1)

                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(gvars, gcoef)],
                        senses=["L"],
                        rhs=[float(cap_vp[0])])



        # pickup_load3(r,i)$(ord(i)<nc+1)..sum((vp,p)$(ord(p)>(nc)),a(p,r)*g(i,vp,p,r))=e=sum((j)$(ord(j)>nc),b(j,r)*dem(j)*z(j,i));

        for r in prod_set:
            for i in cros_set:
                gvars = []
                gcoef = []
                for vp in range(num_sup_v[0]):
                    for p in range(len(sup_set)):
                        gvars.append(g[i - 1][p + len(cros_set) - 1][r - 1][vp])
                        gcoef.append(float(a.values[p - 1][r - 1]))

                zvars = []
                zcoef = []

                for j in range(len(del_set)):
                    zvars.append(z[j][i - 1])
                    zcoef.append(-float(b.values[j][r - 1]) * float(dem.values[j]))

                allvars = []
                allvarscoef = []
                allvars = gvars + zvars
                allvarscoef = gcoef + zcoef
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(allvars, allvarscoef)],
                    senses=["E"],
                    rhs=[0])


        # pickup_load4(i,vp,t,p)$((ord(i)<nc+1) and (ord(t)>nc))..(1-y(i,vp,t,p))*cap_vp+s(i,vp,t,p)=g=sum(r,g(i,vp,t,r))+sum(pp,s(i,vp,pp,t));

        for i in cros_set:

            for t in sup_set:
                for vp in range(num_sup_v[0]):

                    svars2 = []
                    scoefs2 = []
                    for pp in range(len(cros_sup_set)):
                        if pp != (t - len(cros_del_set) + len(cros_set) - 1):
                            svars2.append(s[i - 1][pp][t - len(cros_del_set) + len(cros_set) - 1][vp])
                            scoefs2.append(1)
                    gvars = []
                    gcoefs = []
                    for r in prod_set:
                        gvars.append(g[i - 1][t - len(cros_del_set) + len(cros_set) - 1][r - 1][vp])
                        gcoefs.append(1)

                    for p in range(len(cros_sup_set)):
                        if p != (t - len(cros_del_set) + len(cros_set) - 1):
                            yvars = []
                            ycoefs = []
                            yvars.append(y[i - 1][t - len(cros_del_set) + len(cros_set) - 1][p][vp])
                            # 0.001 is for modifying such that s forced to be zero when we don't need meet some suppliers
                            ycoefs.append(float(cap_vp[0]))

                            svars = []
                            scoefs = []
                            svars.append(s[i - 1][t - len(cros_del_set) + len(cros_set) - 1][p][vp])
                            scoefs.append(-1)

                            allvars = gvars + svars2 + svars + yvars
                            allvarscoefs = gcoefs + scoefs2 + scoefs + ycoefs

                            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(allvars, allvarscoefs)],
                                senses=["L"],
                                rhs=[float(cap_vp[0])])

        # pickup_load5(i,vp,t,p)$(ord(i)<(nc+1))..s(i,vp,p,t)=l=y(i,vp,p,t)*cap_vp;

        for i in cros_set:
            for t in range(len(cros_sup_set)):
                for vp in range(num_sup_v[0]):
                    for p in range(len(cros_sup_set)):
                        if p != t:
                            yvars = []
                            ycoefs = []
                            yvars.append(y[i - 1][p][t][vp])
                            ycoefs.append(-float(cap_vp[0]))

                            svars = []
                            scoefs = []
                            svars.append(s[i - 1][p][t][vp])
                            scoefs.append(1)

                            allvars = svars + yvars
                            allvarscoefs = scoefs + ycoefs

                            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(allvars, allvarscoefs)],
                                senses=["L"],
                                rhs=[0])


        # DirectShipment(i)$(ord(i)>nc)..sum((j)$(ord(j)<(nc+1)),z(i,j))+ sum((p)$(ord(p)>(nc)),sum(r,b(i,r)*a(p,r))*D(p,i))=e=1;

        for i in range(len(del_set)):

            zvars = []
            zcoefs = []
            for j in range(len(cros_set)):
                zvars.append(z[i][j])
                zcoefs.append(1)

            Dvars = []
            Dcoefs = []
            for p in range(len(sup_set)):

                Dvars.append(D[p][i])
                coef = 0
                for r in range(len(prod_set)):
                    coef += float(b.values[i][r]) * float(a.values[p][r])

                Dcoefs.append(coef)

            allvars = zvars + Dvars
            allvarscoefs = zcoefs + Dcoefs
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(allvars, allvarscoefs)],
                senses=["E"],
                rhs=[1])

        # supplier_capacity(p)$(ord(p)>(nc))..  sum((i,vp,r)$(ord(i)<(nc+1)),g(i,vp,p,r))+sum((i)$(ord(i)>nc),D(p,i)*dem(i))=l=cap_s(p);
        for p in range(len(sup_set)):

            Dvars = []
            Dcoefs = []

            gvars = []
            gcoefs = []
            for i1 in range(len(del_set)):
                Dvars.append(D[p][i1])
                Dcoefs.append(float(dem.values[i1]))

            for i2 in range(len(cros_set)):

                for r in range(len(prod_set)):

                    for vp in range(num_sup_v[0]):
                        gvars.append(g[i2][p + len(cros_set)][r][vp])
                        gcoefs.append(1)

            allvars = gvars + Dvars
            allvarscoefs = gcoefs + Dcoefs
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(allvars, allvarscoefs)],
                senses=["L"],
                rhs=[float(cap_s.values[p])])




        # endregion

        # region Solving the problem
        #problem.parameters.parallel.set(-1)  # opportunistic parallel search mode
        #problem.write("SupplyChain.lp")
        problem.parameters.threads.set(16)

        mastervalue = problem.long_annotations.benders_mastervalue
        idx = problem.long_annotations.add(
            name=problem.long_annotations.benders_annotation,
            defval=mastervalue)
        objtype = problem.long_annotations.object_type.variable

        subone = uSET + gSET + sSET
        problem.long_annotations.set_values(idx, objtype,
                                            [(subone[j], mastervalue + 1) for j in range(len(subone))])

        start_time = time.time()
        problem.solve()
        elapsed_time = time.time() - start_time

        sol = problem.solution
        solution = sol.get_values()
        Optimal_Cost = sol.get_objective_value()
        # endregion

        # region Results
        # print obj value
        print("Optimal cost=" + str(Optimal_Cost))
        # print variables values
        x_bound = (len(del_set) + len(cros_set)) * (len(del_set) + len(cros_set)) * num_del_v[0]
        w_bound = x_bound + len(cros_set)
        z_bound = w_bound + (len(del_set) * len(cros_set))
        u_bound = z_bound + (len(del_set) + len(cros_set)) * (len(del_set) + len(cros_set)) * num_del_v[0]
        D_bound = u_bound + (len(del_set)) * (len(sup_set))
        y_bound = D_bound + (len(cros_set)) * (len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0]
        s_bound = y_bound + (len(cros_set)) * (len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0]

        for k in range(1, len(solution) + 1):

            if k < x_bound and (solution[k - 1] > 0):

                ii = math.ceil(k / ((float(num_del[0]) + float(num_cr[0])) * float(num_del_v[0])))
                modj = k % ((float(num_del[0]) + float(num_cr[0])) * float(num_del_v[0]))
                if modj > 0:
                    jj = math.ceil(modj / num_del_v[0])
                else:
                    jj = float(num_del[0] + num_cr[0])

                modv = modj % num_del_v[0]

                if modv > 0:
                    vv = modv
                else:
                    vv = float(num_del_v[0])
                print("x(" + str(ii) + "," + str(jj) + "," + str(vv) + ")= " + str(solution[k - 1]))

            elif (k > x_bound) and (k < w_bound + 1) and (solution[k - 1] > 0):
                ww = k - x_bound
                print("w(" + str(ww) + ")= " + str(solution[k - 1]))

            elif (k > w_bound) and (k < z_bound + 1) and (solution[k - 1] > 0):
                zz = k - w_bound
                ii = math.ceil(zz / num_cr[0])

                modj = zz % (num_cr[0])
                if modj > 0:
                    jj = modj
                else:
                    jj = float(num_cr[0])
                print("z(" + str(del_set[ii - 1]) + "," + str(jj) + ")= " + str(solution[k - 1]))

            elif k > z_bound and (k < u_bound + 1) and (solution[k - 1] > 0):

                uu = k - z_bound
                ii = math.ceil(uu / ((float(num_del[0]) + float(num_cr[0])) * float(num_del_v[0])))
                modj = uu % ((float(num_del[0]) + float(num_cr[0])) * float(num_del_v[0]))
                if modj > 0:
                    jj = math.ceil(modj / num_del_v[0])
                else:
                    jj = float(num_del[0] + num_cr[0])

                modv = modj % num_del_v[0]

                if modv > 0:
                    vv = modv
                else:
                    vv = float(num_del_v[0])
                print("u(" + str(ii) + "," + str(jj) + "," + str(vv) + ")= " + str(solution[k - 1]))

            elif k > u_bound and (k < D_bound + 1) and (solution[k - 1] > 0):
                DD = k - u_bound
                pp = math.ceil(DD / len(del_set))
                modi = DD % (len(del_set))
                if modi > 0:
                    ii = modi
                else:
                    ii = len(del_set)
                print("D(" + str(pp) + "," + str(ii+len(cros_set)) + ")= " + str(solution[k - 1]))

            elif k > D_bound and (k < y_bound + 1) and (solution[k - 1] > 0):

                yy = k - D_bound
                ii = math.ceil(yy / ((len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0]))
                modt = yy % ((len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0])
                if modt > 0:
                    tt = math.ceil(modt / ((len(cros_sup_set)) * num_sup_v[0]))

                else:
                    tt = len(cros_sup_set)

                modp = modt % ((len(cros_sup_set)) * num_sup_v[0])

                if modp > 0:
                    pp = math.ceil(modp / num_sup_v[0])

                else:
                    pp = len(cros_sup_set)

                modv = modp % (num_sup_v[0])

                if modv > 0:
                    vv = modv

                else:
                    vv = float(num_sup_v[0])
                print("y(" + str(ii) + "," + str(cros_sup_set[tt - 1]) + "," + str(cros_sup_set[pp - 1]) + "," + str(
                    vv) + ")= " + str(solution[k - 1]))


            elif k > y_bound and (k < s_bound + 1) and (solution[k - 1] > 0):
                ss = k - y_bound
                ii = math.ceil(ss / ((len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0]))
                modt = ss % ((len(cros_sup_set)) * (len(cros_sup_set)) * num_sup_v[0])
                if modt > 0:
                    tt = math.ceil(modt / ((len(cros_sup_set)) * num_sup_v[0]))

                else:
                    tt = len(cros_sup_set)

                modp = modt % ((len(cros_sup_set)) * num_sup_v[0])

                if modp > 0:
                    pp = math.ceil(modp / num_sup_v[0])

                else:
                    pp = len(cros_sup_set)

                modv = modp % (num_sup_v[0])

                if modv > 0:
                    vv = modv

                else:
                    vv = float(num_sup_v[0])
                print("s(" + str(ii) + "," + str(cros_sup_set[tt - 1]) + "," + str(cros_sup_set[pp - 1]) + "," + str(
                    vv) + ")= " + str(solution[k - 1]))

            elif k > s_bound and (solution[k - 1] > 0):

                # and (solution[k - 1] > 0)
                gg = k - s_bound

                ii = math.ceil(gg / ((len(cros_sup_set)) * (len(prod_set)) * float(num_sup_v[0])))

                modp = gg % ((len(cros_sup_set)) * (len(prod_set)) * float(num_sup_v[0]))
                if modp > 0:
                    pp = math.ceil(modp / ((len(prod_set)) * float(num_sup_v[0])))

                else:
                    pp = len(cros_sup_set)

                modr = modp % ((len(prod_set)) * float(num_sup_v[0]))

                if modr > 0:
                    rr = math.ceil(modr / (float(num_sup_v[0])))

                else:
                    rr = len(prod_set)

                modv = modr % (float(num_prod[0]))

                if modv > 0:
                    vv = modv
                else:
                    vv = float(num_sup_v[0])

                print("g(" + str(ii) + "," + str(cros_sup_set[pp - 1]) + "," + str(rr) + "," + str(vv) + ")= " + str(
                    solution[k - 1]))


        #problem.write("SupplyChain.lp")
        print("Objective function=",problem.solution.get_objective_value())
        # endregion

        # region Printing result in excel sheet

        name = str(cc)
        worksheet.write(cc, 0, name)
        worksheet.write(cc, 1, problem.solution.get_objective_value())
        worksheet.write(cc, 2, elapsed_time)

    # endregion

    if __name__ == "__main__":
        setupproblem()

#Close Excel file
workbook.close()