/*
 * ConicalSpacer.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Oct 16 2024 */
public class ConicalSpacer {

        public static Model run() {
                Model model = ModelUtil.create("Model");

                model.modelPath("/media/sf_vm_shared/sindy_conical_ML");

                model.label("ConicalSpacer_sindy_compact.mph");

                model.param().set("Rin", "90 [mm]", "Central conductor radius");
                model.param().group().create("par2");
                model.param("par2").set("P", "0.4 [MPa]", "Gas pressure");
                model.param("par2").set("S", "3.2e7 [1/(m^3*s)]", "Ion-pair generation rate");
                model.param("par2").set("Rec", "6e-13 [m^3/s]", "Recombination coefficient");
                model.param("par2").set("mup", "4.8e-6 [m^2/(V*s)]", "Positive mobility");
                model.param("par2").set("mun", "4.8e-6 [m^2/(V*s)]", "Negative mobility");
                model.param("par2").set("Dp", "1.2e-7[m^2/s]", "Positive diffusion");
                model.param("par2").set("Dn", "1.2e-7[m^2/s]", "Negative diffusion");
                model.param("par2").set("er_epoxy", "6", "Relative permittivity epoxy");
                model.param("par2").set("k_epoxy", "4.2e-17[S/m]", "Electric conductivity epoxy");
                model.param("par2").set("er_SF6", "1.002", "Relative permittivity Sf6");
                model.param("par2").set("k_gas", "1e-19[S/m]", "Electric conductivity SF6");
                model.param().group().create("par3");
                model.param("par3").set("Udc", "500 [kV]", "High voltage value");
                model.param("par3").set("t", "0 [s]");
                model.param("par3").set("Ut", "Udc*min(t/tau,1)");
                model.param("par3").set("TFinal", "8000 [h]", "Final integration time");
                model.param("par3").set("tau", "1 [ms]");
                model.param().label("Geometric Parameters ");
                model.param("par2").label("Physical Parameters");
                model.param("par3").label("Simulation Parameters");

                model.component().create("comp1", true);

                model.component("comp1").geom().create("geom1", 2);

                model.result().table().create("tbl21", "Table");
                model.result().table().create("tbl22", "Table");
                model.result().table().create("tbl23", "Table");
                model.result().table().create("tbl24", "Table");
                model.result().table().create("tbl5", "Table");
                model.result().table().create("tbl7", "Table");
                model.result().table().create("tbl8", "Table");
                model.result().table().create("tbl9", "Table");
                model.result().table().create("tbl10", "Table");
                model.result().table().create("tbl11", "Table");
                model.result().table().create("tbl12", "Table");
                model.result().table().create("tbl13", "Table");
                model.result().table().create("tbl14", "Table");

                model.component("comp1").geom("geom1").axisymmetric(true);

                model.component("comp1").mesh().create("mesh1");

                model.component("comp1").geom("geom1").create("imp1", "Import");
                model.component("comp1").geom("geom1").feature("imp1").set("type", "dxf");
                model.component("comp1").geom("geom1").feature("imp1")
                                .set("filename", "/media/sf_vm_shared/sindy_conical_ML/Disk_Spacer1.dxf");
                model.component("comp1").geom("geom1").feature("imp1").set("alllayers",
                                new String[] { "0", "Defpoints" });
                model.component("comp1").geom("geom1").create("sca1", "Scale");
                model.component("comp1").geom("geom1").feature("sca1").setIndex("factor", "1e-3", 0);
                model.component("comp1").geom("geom1").feature("sca1").selection("input").set("imp1");
                model.component("comp1").geom("geom1").create("mov1", "Move");
                model.component("comp1").geom("geom1").feature("mov1").set("specify", "pos");
                model.component("comp1").geom("geom1").feature("mov1").set("newpos", "coord");
                model.component("comp1").geom("geom1").feature("mov1").set("newposx", "Rin");
                model.component("comp1").geom("geom1").feature("mov1").selection("input").set("sca1");
                model.component("comp1").geom("geom1").feature("mov1").selection("oldposvertex").set("sca1(1)", 1);
                model.component("comp1").geom("geom1").run();
                model.component("comp1").geom("geom1").run("fin");

                model.component("comp1").selection().create("sel1", "Explicit");
                model.component("comp1").selection("sel1").set(1, 3);
                model.component("comp1").selection().create("sel2", "Explicit");
                model.component("comp1").selection("sel2").set(2);
                model.component("comp1").selection().create("sel3", "Explicit");
                model.component("comp1").selection("sel3").geom("geom1", 1);
                model.component("comp1").selection("sel3").set(1, 3, 5);
                model.component("comp1").selection().create("sel4", "Explicit");
                model.component("comp1").selection("sel4").geom("geom1", 1);
                model.component("comp1").selection("sel4").set(12, 13, 14);
                model.component("comp1").selection().create("sel5", "Explicit");
                model.component("comp1").selection("sel5").geom("geom1", 1);
                model.component("comp1").selection("sel5").set(6, 8, 11, 16, 17);
                model.component("comp1").selection().create("sel6", "Explicit");
                model.component("comp1").selection("sel6").geom("geom1", 1);
                model.component("comp1").selection("sel6").set(4, 9, 10, 15, 18);
                model.component("comp1").selection("sel1").label("Gas");
                model.component("comp1").selection("sel2").label("Insulator");
                model.component("comp1").selection("sel3").label("HV electrode");
                model.component("comp1").selection("sel4").label("Ground electrode");
                model.component("comp1").selection("sel5").label("Interface Upper");
                model.component("comp1").selection("sel6").label("Interface Lower");

                model.component("comp1").variable().create("var1");
                model.component("comp1").variable("var1")
                                .set("JGr", "(ec.Er*(mup*cp+mun*cn)-(Dp*cpr-Dn*cnr))*F_const", "Gas current density");
                model.component("comp1").variable("var1").set("JGz", "(ec.Ez*(mup*cp+mun*cn)-(Dp*cpz-Dn*cnz))*F_const");

                model.view().create("view2", 3);
                model.view().create("view4", 3);

                model.component("comp1").material().create("mat1", "Common");
                model.component("comp1").material().create("mat2", "Common");
                model.component("comp1").material("mat1").selection().named("sel1");
                model.component("comp1").material("mat2").selection().named("sel2");

                model.component("comp1").cpl().create("intop1", "Integration");
                model.component("comp1").cpl("intop1").selection().set(1, 3);

                model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
                model.component("comp1").physics("ec").create("bcs1", "BoundaryCurrentSource", 1);
                model.component("comp1").physics("ec").feature("bcs1").selection().named("sel5");
                model.component("comp1").physics("ec").create("bcs2", "BoundaryCurrentSource", 1);
                model.component("comp1").physics("ec").feature("bcs2").selection().named("sel6");
                model.component("comp1").physics("ec").create("pot1", "ElectricPotential", 1);
                model.component("comp1").physics("ec").feature("pot1").selection().named("sel3");
                model.component("comp1").physics("ec").create("gnd1", "Ground", 1);
                model.component("comp1").physics("ec").feature("gnd1").selection().named("sel4");
                model.component("comp1").physics().create("tds", "DilutedSpecies", "geom1");
                model.component("comp1").physics("tds").field("concentration").field("cp");
                model.component("comp1").physics("tds").field("concentration").component(new String[] { "cp" });
                model.component("comp1").physics("tds").selection().named("sel1");
                model.component("comp1").physics("tds").create("reac1", "Reactions", 2);
                model.component("comp1").physics("tds").feature("reac1").selection().named("sel1");
                model.component("comp1").physics("tds").create("open1", "OpenBoundary", 1);
                model.component("comp1").physics("tds").feature("open1").selection().all();
                model.component("comp1").physics("tds").create("conc1", "Concentration", 1);
                model.component("comp1").physics("tds").feature("conc1").selection().named("sel3");
                model.component("comp1").physics().create("tds2", "DilutedSpecies", "geom1");
                model.component("comp1").physics("tds2").field("concentration").field("cn");
                model.component("comp1").physics("tds2").field("concentration").component(new String[] { "cn" });
                model.component("comp1").physics("tds2").selection().named("sel1");
                model.component("comp1").physics("tds2").create("reac1", "Reactions", 2);
                model.component("comp1").physics("tds2").feature("reac1").selection().named("sel1");
                model.component("comp1").physics("tds2").create("open1", "OpenBoundary", 1);
                model.component("comp1").physics("tds2").feature("open1").selection().all();
                model.component("comp1").physics("tds2").create("conc1", "Concentration", 1);
                model.component("comp1").physics("tds2").feature("conc1").selection().named("sel4");

                model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
                model.component("comp1").mesh("mesh1").create("ftri2", "FreeTri");
                model.component("comp1").mesh("mesh1").feature("ftri1").selection().named("sel2");
                model.component("comp1").mesh("mesh1").feature("ftri1").create("size1", "Size");
                model.component("comp1").mesh("mesh1").feature("ftri2").selection().named("sel1");
                model.component("comp1").mesh("mesh1").feature("ftri2").create("size1", "Size");

                model.result().table("tbl21").label("darkcurrent_grd_lineintegration");
                model.result().table("tbl21").comments("dark currents");
                model.result().table("tbl22").label("darkcurrent_hv_lineintegration");
                model.result().table("tbl22").comments("dark currents");
                model.result().table("tbl23").label("darkcurrent_sld_hv_lineintegration");
                model.result().table("tbl23").comments("dark currents");
                model.result().table("tbl24").label("darkcurrent_sld_grd_lineintegration");
                model.result().table("tbl24").comments("dark currents");
                model.result().table("tbl5").label("volumetric_charge");
                model.result().table("tbl5").comments("Surface Integration 2");
                model.result().table("tbl7").label("tds_stats");
                model.result().table("tbl7").comments("tds stats");
                model.result().table("tbl8").label("E_current_avg");
                model.result().table("tbl8").comments("e current avg gas");
                model.result().table("tbl9").label("E_current_max");
                model.result().table("tbl9").comments("e current max gas");
                model.result().table("tbl10").label("E_current_min");
                model.result().table("tbl10").comments("e current min gas");
                model.result().table("tbl11").label("maximum electric field cathode");
                model.result().table("tbl11").comments("Line Maximum 2");
                model.result().table("tbl12").label("maximum electric field anode");
                model.result().table("tbl12").comments("maximum electric field anode");
                model.result().table("tbl13").label("maximum electric field interface 1");
                model.result().table("tbl13").comments("maximum eletric field interface 1");
                model.result().table("tbl14").label("maximum electric field interface 2");
                model.result().table("tbl14").comments("maximum electric field interface 2");

                model.component("comp1").view("view1").axis().set("xmin", -0.2078041434288025);
                model.component("comp1").view("view1").axis().set("xmax", 0.565335750579834);
                model.component("comp1").view("view1").axis().set("ymin", 0.017528152093291283);
                model.component("comp1").view("view1").axis().set("ymax", 0.435023695230484);

                model.component("comp1").material("mat1").propertyGroup("def")
                                .set("electricconductivity", new String[] { "k_gas", "0", "0", "0", "k_gas", "0", "0",
                                                "0", "k_gas" });
                model.component("comp1").material("mat1").propertyGroup("def")
                                .set("relpermittivity", new String[] { "er_SF6", "0", "0", "0", "er_SF6", "0", "0", "0",
                                                "er_SF6" });
                model.component("comp1").material("mat2").propertyGroup("def")
                                .set("electricconductivity",
                                                new String[] { "k_epoxy", "0", "0", "0", "k_epoxy", "0", "0", "0",
                                                                "k_epoxy" });
                model.component("comp1").material("mat2").propertyGroup("def")
                                .set("relpermittivity",
                                                new String[] { "er_epoxy", "0", "0", "0", "er_epoxy", "0", "0", "0",
                                                                "er_epoxy" });

                model.component("comp1").cpl("intop1").set("axisym", true);

                model.component("comp1").physics("ec").feature("bcs1")
                                .set("Qjs", "ec.nr*(down(ec.Jr)-up(JGr))+ec.nz*(down(ec.Jz)-up(JGz))");
                model.component("comp1").physics("ec").feature("bcs2")
                                .set("Qjs", "-ec.nr*(up(ec.Jr)-down(JGr))-ec.nz*(up(ec.Jz)-down(JGz))");
                model.component("comp1").physics("ec").feature("pot1").set("V0", "Ut");
                model.component("comp1").physics("tds").label("Transport of Diluted Species - Positive Ions");
                model.component("comp1").physics("tds").prop("AdvancedSettings").set("ConvectiveTerm", "cons");
                model.component("comp1").physics("tds").feature("sp1").label("Species Properties");
                model.component("comp1").physics("tds").feature("cdm1")
                                .set("u", new String[][] { { "mup*ec.Er" }, { "0" }, { "mup*ec.Ez" } });
                model.component("comp1").physics("tds").feature("cdm1")
                                .set("D_cp", new String[][] { { "Dp" }, { "0" }, { "0" }, { "0" }, { "Dp" }, { "0" },
                                                { "0" }, { "0" },
                                                { "Dp" } });
                model.component("comp1").physics("tds").feature("reac1")
                                .set("R_cp", "S/N_A_const-Rec*N_A_const*max(comp1.cn,0)*max(comp1.cp,0)");
                model.component("comp1").physics("tds").feature("conc1").set("species", true);
                model.component("comp1").physics("tds2").label("Transport of Diluted Species - Negative Ions");
                model.component("comp1").physics("tds2").prop("AdvancedSettings").set("ConvectiveTerm", "cons");
                model.component("comp1").physics("tds2").feature("cdm1")
                                .set("u", new String[][] { { "-mun*ec.Er" }, { "0" }, { "-mun*ec.Ez" } });
                model.component("comp1").physics("tds2").feature("cdm1")
                                .set("D_cn", new String[][] { { "Dn" }, { "0" }, { "0" }, { "0" }, { "Dn" }, { "0" },
                                                { "0" }, { "0" },
                                                { "Dn" } });
                model.component("comp1").physics("tds2").feature("reac1")
                                .set("R_cn", "S/N_A_const-Rec*N_A_const*max(comp1.cn,0)*max(comp1.cp,0)");
                model.component("comp1").physics("tds2").feature("conc1").set("species", true);

                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hauto", 1);
                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("custom", "on");
                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hmax", 0.002);
                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hmaxactive", true);
                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hmin", 1.0E-5);
                model.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hminactive", false);
                model.component("comp1").mesh("mesh1").feature("ftri2").feature("size1").set("hauto", 1);
                model.component("comp1").mesh("mesh1").run();

                model.study().create("std1");
                model.study("std1").create("time", "Transient");

                model.sol().create("sol1");
                model.sol("sol1").study("std1");
                model.sol("sol1").attach("std1");
                model.sol("sol1").create("st1", "StudyStep");
                model.sol("sol1").create("v1", "Variables");
                model.sol("sol1").create("t1", "Time");
                model.sol("sol1").feature("t1").create("fc1", "FullyCoupled");
                model.sol("sol1").feature("t1").create("d1", "Direct");
                model.sol("sol1").feature("t1").create("i1", "Iterative");
                model.sol("sol1").feature("t1").feature("i1").create("mg1", "Multigrid");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").create("sl1", "SORLine");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").create("sl1", "SORLine");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
                model.sol("sol1").feature("t1").feature().remove("fcDef");

                model.result().numerical().create("int11", "IntLine");
                model.result().numerical().create("int12", "IntLine");
                model.result().numerical().create("int13", "IntLine");
                model.result().numerical().create("int14", "IntLine");
                model.result().numerical().create("int2", "IntSurface");
                model.result().numerical().create("int3", "IntSurface");
                model.result().numerical().create("av1", "AvSurface");
                model.result().numerical().create("max1", "MaxSurface");
                model.result().numerical().create("min1", "MinSurface");
                model.result().numerical().create("max2", "MaxLine");
                model.result().numerical().create("max3", "MaxLine");
                model.result().numerical().create("max4", "MaxLine");
                model.result().numerical().create("max5", "MaxLine");
                model.result().numerical("int11").selection().set(12, 14);
                model.result().numerical("int12").selection().set(1, 5);
                model.result().numerical("int13").selection().set(3);
                model.result().numerical("int14").selection().set(13);
                model.result().numerical("int2").selection().set(1, 3);
                model.result().numerical("int3").selection().set(1, 3);
                model.result().numerical("av1").selection().set(1, 3);
                model.result().numerical("max1").selection().set(1, 3);
                model.result().numerical("min1").selection().set(1, 3);
                model.result().numerical("max2").selection().set(12, 13, 14);
                model.result().numerical("max3").selection().set(1, 3, 5);
                model.result().numerical("max4").selection().set(6, 8, 11, 16, 17);
                model.result().numerical("max5").selection().set(4, 9, 10, 15, 18);
                // model.result().create("pg2", "PlotGroup2D");
                model.result().create("pg3", "PlotGroup2D");
                model.result().create("pg4", "PlotGroup2D");
                model.result().create("pg6", "PlotGroup2D");
                model.result().create("pg8", "PlotGroup1D");
                model.result().create("pg9", "PlotGroup2D");
                model.result().create("pg10", "PlotGroup1D");
                model.result().create("pg11", "PlotGroup2D");
                // model.result().export().create("plot21", "Plot");
                // model.result().export().create("plot22", "Plot");
                model.result().export().create("plot31", "Plot");
                model.result().export().create("plot32", "Plot");
                model.result().export().create("plot4", "Plot");
                model.result().export().create("plot6", "Plot");
                model.result().export().create("plot62", "Plot");
                model.result().export().create("plot81", "Plot");
                model.result().export().create("plot82", "Plot");
                model.result().export().create("plot83", "Plot");
                model.result().export().create("plot84", "Plot");
                model.result().export().create("plot85", "Plot");
                model.result().export().create("plot86", "Plot");
                model.result().export().create("plot91", "Plot");
                model.result().export().create("plot92", "Plot");
                // model.result().export().create("plot101", "Plot");
                model.result().export().create("plot102", "Plot");
                model.result().export().create("plot103", "Plot");
                model.result().export().create("plot104", "Plot");
                model.result().export().create("plot111", "Plot");
                model.result().export().create("plot112", "Plot");

                // model.result("pg2").selection().geom("geom1", 2);
                // model.result("pg2").selection().set(new int[] { 2 });
                // model.result("pg2").create("surf1", "Surface");
                // model.result("pg2").feature("surf1").set("expr", "ec.Ez");
                // model.result("pg2").create("surf2", "Surface");
                // model.result("pg2").feature("surf2").set("expr", "ec.Er");

                model.result("pg3").selection().geom("geom1", 2);
                model.result("pg3").selection().set(new int[] { 1, 3 });
                model.result("pg3").create("surf1", "Surface");
                model.result("pg3").feature("surf1").set("expr", "ec.Ez");
                model.result("pg3").create("surf2", "Surface");
                model.result("pg3").feature("surf2").set("expr", "ec.Er");

                model.result("pg4").create("surf1", "Surface");
                model.result("pg4").feature("surf1").set("expr", "cp");

                model.result("pg6").create("surf1", "Surface");
                model.result("pg6").feature("surf1").set("expr", "cn");

                model.result("pg8").create("lngr1", "LineGraph");
                model.result("pg8").create("lngr2", "LineGraph");
                model.result("pg8").create("lngr3", "LineGraph");
                model.result("pg8").create("lngr4", "LineGraph");
                model.result("pg8").create("lngr5", "LineGraph");
                model.result("pg8").create("lngr6", "LineGraph");
                model.result("pg8").feature("lngr1").selection().set(new int[] { 6, 8, 11, 16, 17 });
                model.result("pg8").feature("lngr1").set("expr", "ec.rhoqs");
                model.result("pg8").feature("lngr2").selection().set(new int[] { 4, 9, 10, 15, 18 });
                model.result("pg8").feature("lngr2").set("expr", "ec.rhoqs");
                model.result("pg8").feature("lngr3").selection().set(new int[] { 6, 8, 11, 16, 17 });
                model.result("pg8").feature("lngr3").set("expr", "ec.Ez");
                model.result("pg8").feature("lngr4").selection().set(new int[] { 6, 8, 11, 16, 17 });
                model.result("pg8").feature("lngr4").set("expr", "ec.Er");
                model.result("pg8").feature("lngr5").selection().set(new int[] { 4, 9, 10, 15, 18 });
                model.result("pg8").feature("lngr5").set("expr", "ec.Ez");
                model.result("pg8").feature("lngr6").selection().set(new int[] { 4, 9, 10, 15, 18 });
                model.result("pg8").feature("lngr6").set("expr", "ec.Er");

                model.result("pg9").selection().geom("geom1", 2);
                model.result("pg9").selection().set(new int[] { 1, 3 });
                model.result("pg9").create("surf1", "Surface");
                model.result("pg9").feature("surf1").set("expr", "JGz");
                model.result("pg9").create("surf2", "Surface");
                model.result("pg9").feature("surf2").set("expr", "JGr");

                // model.result("pg10").create("lngr1", "LineGraph");
                // model.result("pg10").feature("lngr1").selection().set(new int[] { 5, 1, 3 });
                // model.result("pg10").feature("lngr1").set("expr", "ec.Ez");
                model.result("pg10").create("lngr2", "LineGraph");
                model.result("pg10").feature("lngr2").selection().set(new int[] { 5, 1, 3 });
                model.result("pg10").feature("lngr2").set("expr", "ec.Er");
                model.result("pg10").create("lngr3", "LineGraph");
                model.result("pg10").feature("lngr3").selection().set(new int[] { 12, 13, 14 });
                model.result("pg10").feature("lngr3").set("expr", "ec.Ez");
                model.result("pg10").create("lngr4", "LineGraph");
                model.result("pg10").feature("lngr4").selection().set(new int[] { 12, 13, 14 });
                model.result("pg10").feature("lngr4").set("expr", "ec.Er");

                model.result("pg11").selection().geom("geom1", 2);
                model.result("pg11").selection().set(new int[] { 1, 3 });
                model.result("pg11").create("surf1", "Surface");
                model.result("pg11").feature("surf1").set("expr", "d(ec.Ez, z)");
                model.result("pg11").create("surf2", "Surface");
                model.result("pg11").feature("surf2").set("expr", "d(ec.Er, r)");

                model.study("std1").feature("time").set("tlist", "range(0,(TFinal-0)/999,TFinal)");
                model.study("std1").feature("time").set("usertol", true);
                model.study("std1").feature("time").set("rtol", "1E-5");

                model.sol("sol1").attach("std1");
                model.sol("sol1").feature("st1").label("Compile Equations: Time Dependent");
                model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
                model.sol("sol1").feature("v1")
                                .set("clist", new String[] { "{range(0,(TFinal-0)/999,TFinal)}[s]",
                                                "3600.0000000000005[s]" });
                model.sol("sol1").feature("t1").label("Time-Dependent Solver 1.1");
                model.sol("sol1").feature("t1").set("tlist", "range(0,(TFinal-0)/999,TFinal)");
                model.sol("sol1").feature("t1").set("rtol", "1E-5");
                model.sol("sol1").feature("t1").set("maxorder", 2);
                model.sol("sol1").feature("t1").set("stabcntrl", true);
                model.sol("sol1").feature("t1").feature("dDef").label("Direct 2");
                model.sol("sol1").feature("t1").feature("aDef").label("Advanced 1");
                model.sol("sol1").feature("t1").feature("aDef").set("cachepattern", true);
                model.sol("sol1").feature("t1").feature("fc1").label("Fully Coupled 1.1");
                model.sol("sol1").feature("t1").feature("fc1").set("linsolver", "d1");
                model.sol("sol1").feature("t1").feature("fc1").set("maxiter", 8);
                model.sol("sol1").feature("t1").feature("fc1").set("damp", "0.9");
                model.sol("sol1").feature("t1").feature("fc1").set("jtech", "onevery");
                model.sol("sol1").feature("t1").feature("fc1").set("stabacc", "aacc");
                model.sol("sol1").feature("t1").feature("fc1").set("aaccdim", 5);
                model.sol("sol1").feature("t1").feature("fc1").set("aaccmix", 0.9);
                model.sol("sol1").feature("t1").feature("d1").label("Direct (Merged)");
                model.sol("sol1").feature("t1").feature("i1").label("AMG, concentrations (tds2)");
                model.sol("sol1").feature("t1").feature("i1").set("maxlinit", 50);
                model.sol("sol1").feature("t1").feature("i1").feature("ilDef").label("Incomplete LU 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").label("Multigrid 1.1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").set("prefun", "saamg");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").set("maxcoarsedof", 50000);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").set("saamgcompwise", true);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").set("usesmooth", false);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").label("Presmoother 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("soDef")
                                .label("SOR 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("sl1")
                                .label("SOR Line 1.1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("sl1")
                                .set("linesweeptype", "ssor");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("sl1").set("iter",
                                1);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("sl1")
                                .set("linerelax", 0.7);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("pr").feature("sl1").set("relax",
                                0.5);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").label("Postsmoother 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("soDef")
                                .label("SOR 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("sl1")
                                .label("SOR Line 1.1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("sl1")
                                .set("linesweeptype", "ssor");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("sl1").set("iter",
                                1);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("sl1")
                                .set("linerelax", 0.7);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("po").feature("sl1").set("relax",
                                0.5);
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").label("Coarse Solver 1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").feature("dDef")
                                .label("Direct 2");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1")
                                .label("Direct 1.1");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1")
                                .set("linsolver", "pardiso");
                model.sol("sol1").feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1")
                                .set("pivotperturb", 1.0E-13);

                model.result().numerical("int11").label("dark currents_grd");
                model.result().numerical("int11").set("table", "tbl21");
                model.result().numerical("int11").set("expr", new String[] { "JGr*nr+JGz*nz" });
                model.result().numerical("int11").set("unit", new String[] { "A" });
                model.result().numerical("int11").set("descr", new String[] { "Dependent variable u" });
                model.result().numerical("int11").set("intsurface", true);
                model.result().numerical("int12").label("dark currents_hv");
                model.result().numerical("int12").set("table", "tbl22");
                model.result().numerical("int12").set("expr", new String[] { "JGr*nr+JGz*nz" });
                model.result().numerical("int12").set("unit", new String[] { "A" });
                model.result().numerical("int12").set("descr", new String[] { "Dependent variable u" });
                model.result().numerical("int12").set("intsurface", true);
                model.result().numerical("int13").label("dark currents_sld_hv");
                model.result().numerical("int13").set("table", "tbl23");
                model.result().numerical("int13").set("expr", new String[] { "ec.Jr*ec.nr+ec.Jz*ec.nz" });
                model.result().numerical("int13").set("unit", new String[] { "A" });
                model.result().numerical("int13").set("descr", new String[] { "Dependent variable u" });
                model.result().numerical("int13").set("intsurface", true);
                model.result().numerical("int14").label("dark currents_sld_grd");
                model.result().numerical("int14").set("table", "tbl24");
                model.result().numerical("int14").set("expr", new String[] { "ec.Jr*ec.nr+ec.Jz*ec.nz" });
                model.result().numerical("int14").set("unit", new String[] { "A" });
                model.result().numerical("int14").set("descr", new String[] { "Dependent variable u" });
                model.result().numerical("int14").set("intsurface", true);
                model.result().numerical("int2").label("volumetric charge");
                model.result().numerical("int2").set("table", "tbl5");
                model.result().numerical("int2").set("expr", new String[] { "ec.rhoq" });
                model.result().numerical("int2").set("unit", new String[] { "C" });
                model.result().numerical("int2").set("descr", new String[] { "Integration 3" });
                model.result().numerical("int2").set("intvolume", true);
                model.result().numerical("int3").label("tds stats");
                model.result().numerical("int3").set("table", "tbl7");
                model.result().numerical("int3")
                                .set("expr", new String[] { "cp", "cn", "tds.R_cp",
                                                "Rec*N_A_const*max(cn, 0)*max(cp,0)" });
                model.result().numerical("int3").set("unit", new String[] { "mol", "mol", "mol/s", "mol/s" });
                model.result().numerical("int3")
                                .set("descr", new String[] { "Concentration positive", "Concentration negative",
                                                "Generation",
                                                "Recombination" });
                model.result().numerical("int3").set("intvolume", true);
                model.result().numerical("av1").label("e current avg gas");
                model.result().numerical("av1").set("table", "tbl8");
                model.result().numerical("av1")
                                .set("expr",
                                                new String[] { "ec.normE", "ec.normJ", "sqrt(JGz^2+JGr^2)",
                                                                "h*sqrt(tds.u^2+tds.w^2)/Dp" });
                model.result().numerical("av1").set("unit", new String[] { "V/m", "A/m^2", "A/m^2", "1" });
                model.result().numerical("av1")
                                .set("descr", new String[] { "Electric field norm", "Current density norm",
                                                "Gas current density",
                                                "Peclet" });
                model.result().numerical("av1").set("intvolume", true);
                model.result().numerical("max1").label("e current max gas");
                model.result().numerical("max1").set("table", "tbl9");
                model.result().numerical("max1")
                                .set("expr",
                                                new String[] { "ec.normE", "ec.normJ", "sqrt(JGz^2+JGr^2)",
                                                                "h*sqrt(tds.u^2+tds.w^2)/Dp" });
                model.result().numerical("max1").set("unit", new String[] { "V/m", "A/m^2", "A/m^2", "1" });
                model.result().numerical("max1")
                                .set("descr", new String[] { "Electric field norm", "Current density norm", "",
                                                "Peclet" });
                model.result().numerical("min1").label("e current min gas");
                model.result().numerical("min1").set("table", "tbl10");
                model.result().numerical("min1")
                                .set("expr",
                                                new String[] { "ec.normE", "ec.normJ", "sqrt(JGz^2+JGr^2)",
                                                                "h*sqrt(tds.u^2+tds.w^2)/Dp" });
                model.result().numerical("min1").set("unit", new String[] { "V/m", "A/m^2", "A/m^2", "1" });
                model.result().numerical("min1").set("descr", new String[] { "", "", "", "Peclet" });
                model.result().numerical("max2").label("maximum electric field cathode");
                model.result().numerical("max2").set("table", "tbl11");
                model.result().numerical("max2").set("expr", new String[] { "ec.normE", "ec.Er", "ec.Ez" });
                model.result().numerical("max2").set("unit", new String[] { "V/m", "V/m", "V/m" });
                model.result().numerical("max2")
                                .set("descr", new String[] { "Electric field norm", "Electric field, r-component",
                                                "Electric field, z-component" });
                model.result().numerical("max3").label("maximum electric field anode");
                model.result().numerical("max3").set("table", "tbl12");
                model.result().numerical("max3").set("expr", new String[] { "ec.normE", "ec.Er", "ec.Ez" });
                model.result().numerical("max3").set("unit", new String[] { "V/m", "V/m", "V/m" });
                model.result().numerical("max3")
                                .set("descr", new String[] { "Electric field norm", "Electric field, r-component",
                                                "Electric field, z-component" });
                model.result().numerical("max4").label("maximum electric field interface 1");
                model.result().numerical("max4").set("table", "tbl13");
                model.result().numerical("max4").set("expr", new String[] { "ec.normE", "ec.Er", "ec.Ez" });
                model.result().numerical("max4").set("unit", new String[] { "V/m", "V/m", "V/m" });
                model.result().numerical("max4")
                                .set("descr", new String[] { "Electric field norm", "Electric field, r-component",
                                                "Electric field, z-component" });
                model.result().numerical("max5").label("maximum electric field interface 2");
                model.result().numerical("max5").set("table", "tbl14");
                model.result().numerical("max5").set("expr", new String[] { "ec.normE", "ec.Er", "ec.Ez" });
                model.result().numerical("max5").set("unit", new String[] { "V/m", "V/m", "V/m" });
                model.result().numerical("max5")
                                .set("descr", new String[] { "Electric field norm", "Electric field, r-component",
                                                "Electric field, z-component" });
                model.result().numerical("int11").setResult();
                model.result().numerical("int12").setResult();
                model.result().numerical("int13").setResult();
                model.result().numerical("int14").setResult();
                model.result().numerical("int2").setResult();
                model.result().numerical("int3").setResult();
                model.result().numerical("av1").setResult();
                model.result().numerical("max1").setResult();
                model.result().numerical("min1").setResult();
                model.result().numerical("max2").setResult();
                model.result().numerical("max3").setResult();
                model.result().numerical("max4").setResult();
                model.result().numerical("max5").setResult();
                // model.result("pg2").set("frametype", "spatial");
                // model.result("pg2").feature("surf1").set("resolution", "finer");
                // model.result("pg2").feature("surf2").set("resolution", "finer");
                // model.result().export("plot21").set("plotgroup", "pg2");
                // model.result().export("plot21").set("plot", "surf1");
                // model.result().export("plot22").set("plotgroup", "pg2");
                // model.result().export("plot22").set("plot", "surf2");
                model.result("pg3").set("frametype", "spatial");
                model.result("pg3").feature("surf1").set("resolution", "finer");
                model.result("pg3").feature("surf2").set("resolution", "finer");
                model.result().export("plot31").set("plotgroup", "pg3");
                model.result().export("plot31").set("plot", "surf1");
                model.result().export("plot32").set("plotgroup", "pg3");
                model.result().export("plot32").set("plot", "surf2");

                model.result("pg11").set("frametype", "spatial");
                model.result("pg11").feature("surf1").set("resolution", "finer");
                model.result("pg11").feature("surf2").set("resolution", "finer");
                model.result().export("plot111").set("plotgroup", "pg11");
                model.result().export("plot111").set("plot", "surf1");
                model.result().export("plot112").set("plotgroup", "pg11");
                model.result().export("plot112").set("plot", "surf2");
                return model;
        }

        public static Model run2(Model model) {
                model.result("pg4").set("titletype", "custom");
                model.result("pg4").feature("surf1").set("resolution", "finer");
                model.result().export("plot4").set("plotgroup", "pg4");
                model.result().export("plot4").set("plot", "surf1");
                model.result("pg6").set("titletype", "custom");
                model.result("pg6").feature("surf1").set("resolution", "finer");
                model.result().export("plot6").set("plotgroup", "pg6");
                model.result().export("plot6").set("plot", "surf1");
                model.result("pg8").label("surface charge density");
                model.result("pg8").set("xlabel", "Arc length (m)");
                model.result("pg8").set("ylabel", "Surface charge density lower (C/m<sup>2</sup>)");
                model.result("pg8").feature("lngr1").label("surface charge density superior");
                model.result("pg8").feature("lngr1").set("descractive", true);
                model.result("pg8").feature("lngr1").set("resolution", "finer");
                model.result("pg8").feature("lngr2").label("surface charge density lower");
                model.result("pg8").feature("lngr2").set("descractive", true);
                model.result("pg8").feature("lngr2").set("descr", "Surface charge density lower");
                model.result("pg8").feature("lngr2").set("resolution", "finer");
                model.result("pg8").feature("lngr3").label("Ez superior");
                model.result("pg8").feature("lngr3").set("descractive", true);
                model.result("pg8").feature("lngr3").set("descr", "Ez superior");
                model.result("pg8").feature("lngr3").set("resolution", "finer");
                model.result("pg8").feature("lngr4").label("Er superior");
                model.result("pg8").feature("lngr4").set("descractive", true);
                model.result("pg8").feature("lngr4").set("descr", "Er superior");
                model.result("pg8").feature("lngr4").set("resolution", "finer");
                model.result("pg8").feature("lngr5").label("Ez lower");
                model.result("pg8").feature("lngr5").set("descractive", true);
                model.result("pg8").feature("lngr5").set("descr", "Ez lower");
                model.result("pg8").feature("lngr5").set("resolution", "finer");
                model.result("pg8").feature("lngr6").label("Er lower");
                model.result("pg8").feature("lngr6").set("descractive", true);
                model.result("pg8").feature("lngr6").set("descr", "Er lower");
                model.result("pg8").feature("lngr6").set("resolution", "finer");
                model.result().export("plot81").set("plotgroup", "pg8");
                model.result().export("plot81").set("plot", "lngr1");
                model.result().export("plot82").set("plotgroup", "pg8");
                model.result().export("plot82").set("plot", "lngr2");
                model.result().export("plot83").set("plotgroup", "pg8");
                model.result().export("plot83").set("plot", "lngr3");
                model.result().export("plot84").set("plotgroup", "pg8");
                model.result().export("plot84").set("plot", "lngr4");
                model.result().export("plot85").set("plotgroup", "pg8");
                model.result().export("plot85").set("plot", "lngr5");
                model.result().export("plot86").set("plotgroup", "pg8");
                model.result().export("plot86").set("plot", "lngr6");

                model.result("pg9").feature("surf1").label("current norm");
                model.result("pg9").feature("surf1").set("resolution", "finer");
                model.result("pg9").feature("surf2").set("resolution", "finer");
                model.result().export("plot91").set("plotgroup", "pg9");
                model.result().export("plot91").set("plot", "surf1");
                model.result().export("plot92").set("plotgroup", "pg9");
                model.result().export("plot92").set("plot", "surf2");

                model.result("pg10").label("electrodes electric field");
                model.result("pg10").set("xlabel", "Arc length (m)");
                model.result("pg10").set("ylabel", "Electric field (V/m)");
                // model.result("pg10").feature("lngr1").label("Ez HV");
                // model.result("pg10").feature("lngr1").set("descractive", true);
                // model.result("pg10").feature("lngr1").set("resolution", "finer");
                model.result("pg10").feature("lngr2").label("Er HV");
                model.result("pg10").feature("lngr2").set("descractive", true);
                model.result("pg10").feature("lngr2").set("descr", "Er HV");
                model.result("pg10").feature("lngr2").set("resolution", "finer");
                model.result("pg10").feature("lngr3").label("Ez Grd");
                model.result("pg10").feature("lngr3").set("descractive", true);
                model.result("pg10").feature("lngr3").set("descr", "Ez Grd");
                model.result("pg10").feature("lngr3").set("resolution", "finer");
                model.result("pg10").feature("lngr4").label("Er Grd");
                model.result("pg10").feature("lngr4").set("descractive", true);
                model.result("pg10").feature("lngr4").set("descr", "Er Grd");
                model.result("pg10").feature("lngr4").set("resolution", "finer");

                // model.result().export("plot101").set("plotgroup", "pg10");
                // model.result().export("plot101").set("plot", "lngr1");
                model.result().export("plot102").set("plotgroup", "pg10");
                model.result().export("plot102").set("plot", "lngr2");
                model.result().export("plot103").set("plotgroup", "pg10");
                model.result().export("plot103").set("plot", "lngr3");
                model.result().export("plot104").set("plotgroup", "pg10");
                model.result().export("plot104").set("plot", "lngr4");
                return model;
        }

        public static void run3(Model model, String S, String V) {
                String s_value = S + " [1/(m^3*s)]";
                String v_value = V + " [kV]";
                model.param("par2").set("S", s_value, "Ion-pair generation rate");
                model.param("par3").set("Udc", v_value, "High voltage value");
                model.sol("sol1").runAll();
        }

        public static void save(Model model, String S, String V) {
                double variable1 = Double.parseDouble(S) / 1000;
                double variable2 = Double.parseDouble(V);
                String fileName_1 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceChargeUp_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_2 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceChargeDown_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_3 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceEzUp_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_4 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceErUp_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_5 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceEzDown_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_6 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/surfaceErDown_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_7 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_volumeEz_lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_8 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_volumeEr__lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_9 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/cp_lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_10 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/cn_lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                // String fileName_11 = String.format(
                // "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/solid_volumeEz_lastt_S%.0f_V%.0f.txt",
                // variable1, variable2);
                // String fileName_12 = String.format(
                // "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/solid_volumeEr__lastt_S%.0f_V%.0f.txt",
                // variable1, variable2);
                String fileName_13 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_currentJGz__lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_14 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_currentJGr__lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                // String fileName_15 = String.format(
                // "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/HV_Ez_S%.0f_V%.0f.txt",
                // variable1, variable2);
                String fileName_16 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/HV_Er_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_17 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/Grd_Ez_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_18 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/Grd_Er_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_19 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_volumederEz_lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                String fileName_20 = String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/gas_volumederEr_lastt_S%.0f_V%.0f.txt",
                                variable1, variable2);
                // Use the constructed file name in the export settings
                // model.result().export("plot21").set("filename", fileName_11);
                // model.result().export("plot21").run();
                // model.result().export("plot22").set("filename", fileName_12);
                // model.result().export("plot22").run();
                model.result().export("plot31").set("filename", fileName_7);
                model.result().export("plot31").run();
                model.result().export("plot32").set("filename", fileName_8);
                model.result().export("plot32").run();
                model.result().export("plot4").set("filename", fileName_9);
                model.result().export("plot4").run();
                model.result().export("plot6").set("filename", fileName_10);
                model.result().export("plot6").run();
                model.result().export("plot81").set("filename", fileName_1);
                model.result().export("plot81").set("multiplecurves", "ascolumns");
                model.result().export("plot81").run();
                model.result().export("plot82").set("filename", fileName_2);
                model.result().export("plot82").set("multiplecurves", "ascolumns");
                model.result().export("plot82").run();
                model.result().export("plot83").set("filename", fileName_3);
                model.result().export("plot83").set("multiplecurves", "ascolumns");
                model.result().export("plot83").run();
                model.result().export("plot84").set("filename", fileName_4);
                model.result().export("plot84").set("multiplecurves", "ascolumns");
                model.result().export("plot84").run();
                model.result().export("plot85").set("filename", fileName_5);
                model.result().export("plot85").set("multiplecurves", "ascolumns");
                model.result().export("plot85").run();
                model.result().export("plot86").set("filename", fileName_6);
                model.result().export("plot86").set("multiplecurves", "ascolumns");
                model.result().export("plot86").run();
                model.result().export("plot91").set("filename", fileName_13);
                model.result().export("plot91").run();
                model.result().export("plot92").set("filename", fileName_14);
                model.result().export("plot92").run();
                // model.result().export("plot101").set("filename", fileName_15);
                // model.result().export("plot101").set("multiplecurves", "ascolumns");
                // model.result().export("plot101").run();
                model.result().export("plot102").set("filename", fileName_16);
                model.result().export("plot102").set("multiplecurves", "ascolumns");
                model.result().export("plot102").run();
                model.result().export("plot103").set("filename", fileName_17);
                model.result().export("plot103").set("multiplecurves", "ascolumns");
                model.result().export("plot103").run();
                model.result().export("plot104").set("filename", fileName_18);
                model.result().export("plot104").set("multiplecurves", "ascolumns");
                model.result().export("plot104").run();

                model.result().export("plot111").set("filename", fileName_19);
                model.result().export("plot111").run();
                model.result().export("plot112").set("filename", fileName_20);
                model.result().export("plot112").run();
        }

        public static void save_tables(Model model, String S, String V) {
                model.result().numerical("int11").run();
                model.result().numerical("int11").setResult();
                model.result().table("tbl21").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/dark_currents_grd_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("int12").run();
                model.result().numerical("int12").setResult();
                model.result().table("tbl22").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/dark_currents_hv_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("int13").run();
                model.result().numerical("int13").setResult();
                model.result().table("tbl23").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/dark_currents_sld_hv_S%s_V%s.txt",
                                                S,
                                                V));
                model.result().numerical("int14").run();
                model.result().numerical("int14").setResult();
                model.result().table("tbl24").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/dark_currents_sld_grd_S%s_V%s.txt",
                                                S,
                                                V));
                model.result().numerical("int2").run();
                model.result().numerical("int2").setResult();
                model.result().table("tbl5").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/volumetric_charge_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("int3").run();
                model.result().numerical("int3").setResult();
                model.result().table("tbl7").save(
                                String.format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/tds_stats_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("av1").run();
                model.result().numerical("av1").setResult();
                model.result().table("tbl8").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/e_current_avg_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("max1").run();
                model.result().numerical("max1").setResult();
                model.result().table("tbl9").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/e_current_max_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("min1").run();
                model.result().numerical("min1").setResult();
                model.result().table("tbl10").save(String
                                .format("/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/e_current_min_S%s_V%s.txt",
                                                S, V));
                model.result().numerical("max2").run();
                model.result().numerical("max2").setResult();
                model.result().table("tbl11").save(String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/maximum_electric_field_cathode_S%s_V%s.txt",
                                S,
                                V));
                model.result().numerical("max3").run();
                model.result().numerical("max3").setResult();
                model.result().table("tbl12").save(String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/maximum_electric_field_anode_S%s_V%s.txt",
                                S,
                                V));
                model.result().numerical("max4").run();
                model.result().numerical("max4").setResult();
                model.result().table("tbl13").save(String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/maximum_electric_field_upInterface_S%s_V%s.txt",
                                S, V));
                model.result().numerical("max5").run();
                model.result().numerical("max5").setResult();
                model.result().table("tbl14").save(String.format(
                                "/media/sf_vm_shared/sindy_conical_ML/database_comsol_ml/maximum_electric_field_downInterface_S%s_V%s.txt",
                                S, V));
        }

        public static void main(String[] args) {
                Model model = run();
                run2(model);

                // List of S values to iterate over
                String[] sValues = { "8e7", "1e8" }; // DONE "5e7", "1e7", "2e8", "2e7", "4e7"
                // List of V values to iterate over
                String[] vValues = new String[20];
                for (int i = 0; i < 20; i++) {
                        vValues[i] = String.valueOf(15 + i * (500 - 15) / 19);
                        vValues[i] = vValues[i].split("\\.")[0]; // Ensure only integers are passed
                }

                // Iterate over each S value
                for (String S : sValues) {
                        for (String V : vValues) {
                                run3(model, S, V);
                                save(model, S, V);
                                save_tables(model, S, V);
                        }
                }
        }

}
