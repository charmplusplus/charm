#
# spec file for building charm rpm for redhat.
#
Summary: Charm++

%define buildsmp 0
%define charm_options  %{nil}

%if %{buildsmp}
Name: Charm-smp
%define charm_options  %{charm_option} smp
%else
Name: Charm
%endif

Version: 5.9
Release: 12
Copyright: GPL
Group: Applications/System
BuildRoot: /var/tmp/%{name}-root
Source: http://charm.cs.illinois.edu/distrib/Charm-%{version}-%{release}.tar.bz2
URL: http://charm.cs.illinois.edu
Vendor: PPL <charm@cs.illinois.edu>

%description
Charm++ RPM for Linux

%prep
%setup -n %{name}-%{version}-%{release}
bzip2 -dc  %{_sourcedir}/Charm-%{version}-%{release}.tar.bz2 | tar xvf -
%build
./build AMPI net-linux %{charm_options} -O -DCMK_OPTIMIZE
%clean
rm -rf $RPM_BUILD_ROOT
%install
mkdir -p $RPM_BUILD_ROOT
cd tmp; make DESTDIR=$RPM_BUILD_ROOT/usr/local/%{name} install
%post
 
%files
/usr/local/%{name}/*
%doc README LICENSE CHANGES README.cygwin README.win32

