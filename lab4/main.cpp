#include <iostream>
#include <fstream>
#include <omp.h>
#include <math.h>
#include <cstring>
#include <string>
#include "BMPFileRW.h"

using namespace std;

#pragma warning(disable : 4996)
# define M_PI           3.14159265358979323846
typedef double(*TestFunctTemp1)(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize);

double medianLinearFilter(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
            int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;
            for (int dy = -RH; dy <= RH; dy++) {
                int ky = y + dy;

                if (ky < 0)
                    ky = 0;
                if (ky > height - 1)
                    ky = height - 1;

                for (int dx = -RW; dx <= RW; dx++) {
                    int kx = x + dx;

                    if (kx < 0)
                        kx = 0;
                    if (kx > width - 1)
                        kx = width - 1;

                    Summ_Value_R += rgb_in[ky][kx].rgbtRed;
                    Summ_Value_G += rgb_in[ky][kx].rgbtGreen;
                    Summ_Value_B += rgb_in[ky][kx].rgbtBlue;
                }
            }
            LinF_Value_R = Summ_Value_R / size;
            LinF_Value_G = Summ_Value_G / size;
            LinF_Value_B = Summ_Value_B / size;
     
            rgb_out[y][x].rgbtRed = LinF_Value_R;
            rgb_out[y][x].rgbtGreen = LinF_Value_G;
            rgb_out[y][x].rgbtBlue = LinF_Value_B;
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianLinearFilterParallel(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
            int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;
            for (int dy = -RH; dy <= RH; dy++) {
                int ky = y + dy;

                if (ky < 0)
                    ky = 0;
                if (ky > height - 1)
                    ky = height - 1;

                for (int dx = -RW; dx <= RW; dx++) {
                    int kx = x + dx;

                    if (kx < 0)
                        kx = 0;
                    if (kx > width - 1)
                        kx = width - 1;

                    Summ_Value_R += rgb_in[ky][kx].rgbtRed;
                    Summ_Value_G += rgb_in[ky][kx].rgbtGreen;
                    Summ_Value_B += rgb_in[ky][kx].rgbtBlue;
                }
            }
            LinF_Value_R = Summ_Value_R / size;
            LinF_Value_G = Summ_Value_G / size;
            LinF_Value_B = Summ_Value_B / size;

            rgb_out[y][x].rgbtRed = LinF_Value_R;
            rgb_out[y][x].rgbtGreen = LinF_Value_G;
            rgb_out[y][x].rgbtBlue = LinF_Value_B;
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double** gaussMatrix(int size) {
    int RH = size / 2, RW = size / 2;
    double** matrix = new double* [size];

    for (int i = 0; i < size; i++)
        matrix[i] = new double[size];

    double SUM = 0;

    for (int y = -RH; y <= RH; y++) {
        for (int x = -RW; x <= RW; x++) {
            int YK = y + RH;
            int XK = x + RW;
            double CF = (1 / (2 * M_PI * pow(RH, 2))) * exp(-1 * (pow(x, 2) +
                pow(y, 2)) / (2 * pow(RH, 2)));
            matrix[YK][XK] = CF;
            SUM += CF;
        }
    }
    for (int y = -RH; y <= RH; y++) {
        for (int x = -RW; x <= RW; x++) {
            int YK = y + RH;
            int XK = x + RW;
            matrix[YK][XK] /= SUM;
        }
    }
    return matrix;
}

double medianGaussFilter(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    double** matrix = gaussMatrix(ksize);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;

            for (int dy = -RH; dy <= RH; dy++) {
                int ky = y + dy;

                if (ky < 0)
                    ky = 0;
                if (ky > height - 1)
                    ky = height - 1;

                for (int dx = -RW; dx <= RW; dx++) {
                    int kx = x + dx;

                    if (kx < 0)
                        kx = 0;
                    if (kx > width - 1)
                        kx = width - 1;

                    LinF_Value_R += rgb_in[ky][kx].rgbtRed *
                        matrix[dy + RH][dx + RW];
                    LinF_Value_G += rgb_in[ky][kx].rgbtGreen *
                        matrix[dy + RH][dx + RW];
                    LinF_Value_B += rgb_in[ky][kx].rgbtBlue *
                        matrix[dy + RH][dx + RW];
                }
            }
            LinF_Value_R = LinF_Value_R < 0 ? 0 : LinF_Value_R;
            LinF_Value_R = LinF_Value_R > 255 ? 255 : LinF_Value_R;
            rgb_out[y][x].rgbtRed = LinF_Value_R;

            LinF_Value_G = LinF_Value_G < 0 ? 0 : LinF_Value_G;
            LinF_Value_G = LinF_Value_G > 255 ? 255 : LinF_Value_G;
            rgb_out[y][x].rgbtGreen = LinF_Value_G;

            LinF_Value_B = LinF_Value_B < 0 ? 0 : LinF_Value_B;
            LinF_Value_B = LinF_Value_B > 255 ? 255 : LinF_Value_B;
            rgb_out[y][x].rgbtBlue = LinF_Value_B;
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double** gaussMatrixParallel(int size) {
    int RH = size / 2, RW = size / 2;
    double** matrix = new double* [size];

    for (int i = 0; i < size; i++)
        matrix[i] = new double[size];

    double SUM = 0;
#pragma omp parallel for reduction(+:SUM)
    for (int y = -RH; y <= RH; y++) {
        for (int x = -RW; x <= RW; x++) {
            int YK = y + RH;
            int XK = x + RW;
            double CF = (1 / (2 * M_PI * pow(RH, 2))) * exp(-1 * (pow(x, 2) +
                pow(y, 2)) / (2 * pow(RH, 2)));
            matrix[YK][XK] = CF;
            SUM += CF;
        }
    }
#pragma omp parallel for
    for (int y = -RH; y <= RH; y++) {
        for (int x = -RW; x <= RW; x++) {
            int YK = y + RH;
            int XK = x + RW;
            matrix[YK][XK] /= SUM;
        }
    }
    return matrix;
}

double medianGaussFilterParallel(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    double** matrix = gaussMatrixParallel(ksize);

#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;

            for (int dy = -RH; dy <= RH; dy++) {
                int ky = y + dy;

                if (ky < 0)
                    ky = 0;
                if (ky > height - 1)
                    ky = height - 1;

                for (int dx = -RW; dx <= RW; dx++) {
                    int kx = x + dx;

                    if (kx < 0)
                        kx = 0;
                    if (kx > width - 1)
                        kx = width - 1;

                    LinF_Value_R += rgb_in[ky][kx].rgbtRed *
                        matrix[dy + RH][dx + RW];
                    LinF_Value_G += rgb_in[ky][kx].rgbtGreen *
                        matrix[dy + RH][dx + RW];
                    LinF_Value_B += rgb_in[ky][kx].rgbtBlue *
                        matrix[dy + RH][dx + RW];
                }
            }
            LinF_Value_R = LinF_Value_R < 0 ? 0 : LinF_Value_R;
            LinF_Value_R = LinF_Value_R > 255 ? 255 : LinF_Value_R;
            rgb_out[y][x].rgbtRed = LinF_Value_R;

            LinF_Value_G = LinF_Value_G < 0 ? 0 : LinF_Value_G;
            LinF_Value_G = LinF_Value_G > 255 ? 255 : LinF_Value_G;
            rgb_out[y][x].rgbtGreen = LinF_Value_G;

            LinF_Value_B = LinF_Value_B < 0 ? 0 : LinF_Value_B;
            LinF_Value_B = LinF_Value_B > 255 ? 255 : LinF_Value_B;
            rgb_out[y][x].rgbtBlue = LinF_Value_B;
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double medianQuickLinearFilter(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int height, int width, int ksize) {
    double time_start = omp_get_wtime();

    int size = ksize * ksize;
    int RH = ksize / 2, RW = ksize / 2;

    for (int y = 0; y < height; ++y)
    {
        int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;
        for (int x = 0; x < width; ++x)
        {
            int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
       
            if (x == 0) {

                for (int dy = -RH; dy <= RH; dy++) {
                    int ky = y + dy;

                    if (ky < 0)
                        ky = 0;
                    if (ky > height - 1)
                        ky = height - 1;

                    for (int dx = -RW; dx <= RW; dx++) {
                        int kx = x + dx;

                        if (kx < 0)
                            kx = 0;
                        if (kx > width - 1)
                            kx = width - 1;

                        Summ_Value_R += rgb_in[ky][kx].rgbtRed;
                        Summ_Value_G += rgb_in[ky][kx].rgbtGreen;
                        Summ_Value_B += rgb_in[ky][kx].rgbtBlue;
                    }
                }
                LinF_Value_R = Summ_Value_R / size;
                LinF_Value_G = Summ_Value_G / size;
                LinF_Value_B = Summ_Value_B / size;

                rgb_out[y][x].rgbtRed = LinF_Value_R;
                rgb_out[y][x].rgbtGreen = LinF_Value_G;
                rgb_out[y][x].rgbtBlue = LinF_Value_B;

               
            } else if (x > 1) {
                int kx1 = x - RW - 1;
                if (kx1 < 0) {
                    kx1 = 0;
                }
                int kx2 = x + RW;
                if (kx2 > width - 1) {
                    kx2 = width - 1;
                }

                for (int dy = -RH; dy <= RH; dy++) {
                    int ky = y + dy;

                    if (ky < 0)
                        ky = 0;
                    if (ky > height - 1)
                        ky = height - 1;

                    Summ_Value_R -= rgb_in[ky][kx1].rgbtRed;
                    Summ_Value_G -= rgb_in[ky][kx1].rgbtGreen;
                    Summ_Value_B -= rgb_in[ky][kx1].rgbtBlue;
                    Summ_Value_R += rgb_in[ky][kx2].rgbtRed;
                    Summ_Value_G += rgb_in[ky][kx2].rgbtGreen;
                    Summ_Value_B += rgb_in[ky][kx2].rgbtBlue;
                }
                LinF_Value_R = Summ_Value_R / size;
                LinF_Value_G = Summ_Value_G / size;
                LinF_Value_B = Summ_Value_B / size;

                rgb_out[y][x].rgbtRed = LinF_Value_R;
                rgb_out[y][x].rgbtGreen = LinF_Value_G;
                rgb_out[y][x].rgbtBlue = LinF_Value_B;
            }
        }
    }

    double time_stop = omp_get_wtime();
    return time_stop - time_start;
}

double testMedianLinearFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianLinearFilter(rgb_in, rgb_out, height, width, ksize);
}
double testMedianGaussFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianGaussFilter(rgb_in, rgb_out, height, width, ksize);
}
double testMedianQuickLinearFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianQuickLinearFilter(rgb_in, rgb_out, height, width, ksize);
}
double testMedianGaussFilterParallel(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianGaussFilterParallel(rgb_in, rgb_out, height, width, ksize);
}
double testMedianLinearFilterParallel(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
    return medianLinearFilterParallel(rgb_in, rgb_out, height, width, ksize);
}
//double testMedianQuickFilterParallelFor(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int& height, int& width, int& ksize) {
//    return medianQuickFilterParallelFor(rgb_in, rgb_out, height, width, ksize);
//}

char* inBMP(int i) {
    string str;
    str = "c:\\temp\\input_X.bmp";
    char* cstr;
    switch (i)
    {
    case 1:
        str[14] = '1';
        break;
    case 2:
        str[14] = '2';
        break;
    case 3:
        str[14] = '3';
        break;
    case 4:
        str[14] = '4';
        break;
    default:
        break;
    }

    cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    return cstr;
}
char* outBMP(int i, int alg) {
    string str;
    str = "c:\\temp\\input_X_Y.bmp";
    char* cstr;
    switch (i)
    {
    case 1:
        str[14] = '1';
        break;
    case 2:
        str[14] = '2';
        break;
    case 3:
        str[14] = '3';
        break;
    case 4:
        str[14] = '4';
        break;
    default:
        break;
    }

    switch (alg)
    {
    case 0:
        str[16] = '1';
        break;
    case 1:
        str[16] = '2';
        break;
    case 2:
        str[16] = '3';
        break;
    case 3:
        str[16] = '4';
        break;
    case 4:
        str[16] = '5';
        break;
    case 5:
        str[16] = '6';
        break;
    default:
        break;
    }
    cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    cout << cstr << endl;
    return cstr;
}

double AvgTrustedInterval(double& avg, double*& times, int& cnt)
{
    double sd = 0, newAVg = 0;
    int newCnt = 0;
    for (int i = 0; i < cnt; i++)
    {
        sd += (times[i] - avg) * (times[i] - avg);
    }
    sd /= (cnt - 1.0);
    sd = sqrt(sd);
    for (int i = 0; i < cnt; i++)
    {
        if (avg - sd <= times[i] && times[i] <= avg + sd)
        {
            newAVg += times[i];
            newCnt++;
        }
    }
    if (newCnt == 0) newCnt = 1;
    return newAVg / newCnt;
}

double TestIter(void* Funct, RGBTRIPLE** rgb_in, int Height, int Width, int ksize, int iterations, int i, int alg)
{
    double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;;
    double* Times = new double[iterations];
    cout << endl;

    RGBTRIPLE** rgb_out;
    rgb_out = new RGBTRIPLE * [Height];
    rgb_out[0] = new RGBTRIPLE[Width * Height];
    for (int j = 1; j < Height; j++)
    {
        rgb_out[j] = &rgb_out[0][Width * j];
    }

    for (int j = 0; j < iterations; j++)
    {
        curtime = ((*(TestFunctTemp1)Funct)(rgb_in, rgb_out, Height, Width, ksize)) * 1000;
        Times[j] = curtime;
        avgTime += curtime;
        cout << "+";
    }
    cout << endl;

    avgTime /= iterations;
    cout << "AvgTime:" << avgTime << endl;

    avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
    cout << "AvgTimeTrusted:" << avgTimeT << endl;

    char* cstr = outBMP(i, alg);

    BMPWrite(rgb_out, Width, Height, cstr);
    delete[] rgb_out[0];
    delete[] rgb_out;
    delete[] cstr;
    return avgTimeT;
}

void test_functions(void** Functions, string(&function_names)[5])
{
    RGBTRIPLE** rgb_in;
    BITMAPFILEHEADER header;
    BITMAPINFOHEADER bmiHeader;
    int imWidth = 0, imHeight = 0;

    int iters = 2;
    int nd = 0;
    double times[1][5][3][3];
    for (int i = 1; i < 2; i++)
    {
        char* cstr = inBMP(i);
        BMPRead(rgb_in, header, bmiHeader, cstr);
        imWidth = bmiHeader.biWidth;
        imHeight = bmiHeader.biHeight;

        for (int threads = 1; threads <= 4; threads++)
        {
            omp_set_num_threads(threads);
            //перебор алгоритмов по условиям
            for (int alg = 0; alg < 5; alg++)
            {
                int ksize = 7;
                for (int j = 0; j < 1; j++)
                {
                    if (threads == 1)
                    {
                        if (alg == 0 || alg == 1 || alg == 2) {
                            times[nd][alg][j][0] = TestIter(Functions[alg], rgb_in, imHeight, imWidth, ksize, iters, i, alg);
                            // iters - кол-во запусков алгоритма
                            times[nd][alg][j][1] = times[nd][alg][j][0];
                            times[nd][alg][j][2] = times[nd][alg][j][0];
                        }
                    }
                    else
                    {
                        if (alg != 0 && alg != 1 && alg != 2)
                        {
                            times[nd][alg][j][threads - 2] = TestIter(Functions[alg], rgb_in, imHeight, imWidth, ksize, iters, i, alg);
                        }
                    }
                    ksize += 4;
                }
            }
        }
        delete[] cstr;
        nd++;
    }
    ofstream fout("output.txt");
    fout.imbue(locale("Russian"));
    for (int ND = 0; ND < 1; ND++)
    {
        switch (ND)
        {
        case 0:
            cout << "\n----------1280*720----------" << endl;
            break;
        case 1:
            cout << "\n----------1920*1080----------" << endl;
            break;
        case 2:
            cout << "\n----------2580*1080----------" << endl;
            break;
        case 3:
            cout << "\n----------3840*2160----------" << endl;
            break;
        default:
            break;
        }
        for (int alg = 0; alg < 5; alg++)
        {
            for (int threads = 1; threads <= 4; threads++)
            {
                cout << "Поток " << threads << " --------------" << endl;
                for (int j = 0; j < 3; j++)
                {
                    cout << "Ksize = " << j << " --------------" << endl;
                    if (threads == 1)
                    {
                        if (alg == 0 || alg == 1 || alg == 2) {
                            cout << function_names[alg] << "\t" << times[ND][alg][j][0] << " ms." << endl;
                            fout << times[ND][alg][j][0] << endl;
                        }
                    }
                    else
                    {
                        if (alg != 0 && alg != 1 && alg != 2)
                        {
                            cout << function_names[alg] << "\t" << times[ND][alg][j][threads - 2] << " ms." << endl;
                            fout << times[ND][alg][j][threads - 2] << endl;
                        }
                    }
                }
            }
        }
    }
    fout.close();
}

int main()
{
    setlocale(LC_ALL, "RUS");

    void** FunctionsINT = new void* [5]{ testMedianLinearFilter, testMedianGaussFilter, testMedianQuickLinearFilter, medianLinearFilterParallel,
        testMedianGaussFilterParallel };
    string function_names[5]{ "медианная фильтрация(линейная)", "медианная фильтрация(гаус)",
        "медианная фильтрация(быстрая линейная)", "медианная фильтрация(линейный пар)", "медианная фильтрация(гаус пар)" };
    test_functions(FunctionsINT, function_names);



    RGBTRIPLE** rgb_in, ** rgb_out;
    BITMAPFILEHEADER header;
    BITMAPINFOHEADER bmiHeader;
    int imWidth = 0, imHeight = 0;
    BMPRead(rgb_in, header, bmiHeader, "c:\\temp\\input_1.bmp");
    imWidth = bmiHeader.biWidth;
    imHeight = bmiHeader.biHeight;
    std::cout << "Image params:" << imWidth << "x" << imHeight << std::endl;
    rgb_out = new RGBTRIPLE * [imHeight];
    rgb_out[0] = new RGBTRIPLE[imWidth * imHeight];
    for (int i = 1; i < imHeight; i++)
    {
        rgb_out[i] = &rgb_out[0][imWidth * i];
    }

    cout << medianGaussFilter(rgb_in, rgb_out, imHeight, imWidth, 11) << endl;
    BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\test_copy.bmp");
    std::cout << "Image saved\n";

    return 0;
}