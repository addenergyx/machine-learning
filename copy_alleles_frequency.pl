#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

# Example of path to file 
# /warehouse/team229_wh01/lims2_managed_miseq_data/Miseq_026/S143_expSNCA_E35K_4/CRISPResso_on_143_S143_L001_R1_001_143_S143_L001_R2_001/Alleles_frequency_table.txt

GetOptions(
    'path=s' => \my $path,
);

$path = $path || '.';

my @files = `find $path -name "Alleles_frequency_table.txt"`;

if ( scalar(@files) == 0 ){
    die "Can't find Alleles_frequency_table.txt in directory $path";
}

chomp @files; # Removes new line from each filename

foreach my $file (@files) {
    $file =~ s/\.\///;
    (my $name = $file) =~ s/\//_/g;
    $name =~ s/_CRISPResso[A-Za-z0-9_]+\d//g;
    my $dest_path = "./alleles_frequency/$name";
    # when using system commands it's best to use the absolute path of the command and not just cp as cp could be simply aliased to something else
    system("/bin/cp $file $dest_path");
    print "\nFile: $file has been copied\n";
}
